# Dataset + collation for the extracted per-token QKV feature fields.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.extract.tensor_ops import PROJECTIONS
from src.utils.logger import get_logger

logger = get_logger(__name__)

N_CHANNELS = len(PROJECTIONS)
HIDDEN_STATE_CHANNELS = ("H",)


class QKVFieldDataset(Dataset):
    # One example = one folder holding tokens.npy of shape (T, L, M, 3).
    # __getitem__ returns (images, label, origin), images (T, 3, L, M) --
    # channel axis already in conv position (N, C, H, W).

    def __init__(
        self,
        root: str | Path,
        stats: dict | None = None,
        max_tokens: int | None = None,
        origin: str | None = None,
        keep_channels: list[str] | None = None,
        token_buckets: int | None = None,
        layer_permute_seed: int | None = None,
        segment_permute_seed: int | None = None,
        collapse_axis: str | None = None,
    ):
        self.keep_channels = keep_channels
        self.token_buckets = token_buckets
        self.layer_permute_seed = layer_permute_seed
        self.segment_permute_seed = segment_permute_seed
        self.collapse_axis = collapse_axis
        self._layer_perm: torch.Tensor | None = None
        self._segment_perm: torch.Tensor | None = None
        self.root = Path(root)
        manifest = self.root / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(
                f"no manifest at {manifest}. Run `python detector.py extract` first."
            )

        geometry_path = self.root / "geometry.json"
        self.geometry = (
            json.loads(geometry_path.read_text()) if geometry_path.exists() else {}
        )
        # Read THIS corpus's own channel count from geometry.json rather than
        # assuming N_CHANNELS: a hidden-states corpus has 1 channel, not Q/K/V.
        projections = self.geometry.get("projections")
        # n_channels_on_disk is what's in tokens.npy and what normalize() must
        # standardise over (before any channels collapse); n_channels is what
        # the model actually sees (differs only under collapse_axis='channels').
        self.n_channels_on_disk = len(projections) if projections else N_CHANNELS
        self.is_hidden_states = self.n_channels_on_disk == 1
        self.n_channels = 1 if self.collapse_axis == "channels" else self.n_channels_on_disk

        self.records = []
        with open(manifest) as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

        unlabeled = [r for r in self.records if r.get("label", -1) not in (0, 1)]
        if unlabeled:
            raise ValueError(
                f"{len(unlabeled)} examples in {self.root} have no valid label. "
                "Run `python detector.py label` to (re)label them."
            )

        # A response truncated at its very first token leaves n_tokens == 0,
        # which would reach the model as an all-False mask; drop those here.
        empty = [r for r in self.records if r.get("n_tokens", 0) == 0]
        if empty:
            logger.warning(
                "%s: dropping %d example(s) with n_tokens == 0 (empty after "
                "run-on truncation): idx=%s",
                self.root, len(empty), [r["idx"] for r in empty],
            )
            self.records = [r for r in self.records if r.get("n_tokens", 0) > 0]

        self.stats = stats
        self.max_tokens = max_tokens
        self.origin = origin or f"{self.root.parent.name}_{self.root.name}"

    def __len__(self) -> int:
        return len(self.records)

    @property
    def labels(self) -> list[int]:
        return [int(r["label"]) for r in self.records]

    def _load_raw(self, i) -> torch.Tensor:
        # (T, L, M, 3) for example i, un-normalised.
        rec = self.records[i]
        path = self.root / rec["dir"] / "tokens.npy"

        arr = np.load(path)
        field = torch.from_numpy(np.ascontiguousarray(arr)).float()

        if self.max_tokens is not None and field.shape[0] > self.max_tokens:
            field = field[: self.max_tokens]

        if self.token_buckets is not None:
            field = bucket_pool_tokens(field, self.token_buckets)

        return field

    def _finish(self, raw: torch.Tensor) -> torch.Tensor:
        # normalise -> zero unwanted channels -> collapse an axis -> permute layers/segments -> conv position
        field = raw
        if self.stats is not None:
            field = normalize(field, self.stats, n_channels=self.n_channels_on_disk)

        if self.keep_channels is not None:
            if self.is_hidden_states:
                raise ValueError(
                    "model.keep_channels is a Q/K/V-projection ablation and has "
                    "no meaning on a hidden-states corpus (1 channel, not "
                    "named Q/K/V) -- unset it for this data_root."
                )
            field = zero_channels(field, self.keep_channels)

        if self.collapse_axis is not None:
            field = collapse_axis_mean(field, self.collapse_axis)

        if self.layer_permute_seed is not None:
            field = self._permuted_layers(field)

        if self.segment_permute_seed is not None:
            field = self._permuted_segments(field)

        return field.permute(0, 3, 1, 2).contiguous()   # (T, L, M, C) -> (T, C, L, M)

    def _permuted_layers(self, field: torch.Tensor) -> torch.Tensor:
        # One fixed permutation, built on first use and reused for every
        # example/epoch -- a per-call shuffle would leak true adjacency back in.
        n_layers = field.shape[1]
        if self._layer_perm is None or self._layer_perm.shape[0] != n_layers:
            gen = torch.Generator().manual_seed(self.layer_permute_seed)
            self._layer_perm = torch.randperm(n_layers, generator=gen)
        return field[:, self._layer_perm]

    def _permuted_segments(self, field: torch.Tensor) -> torch.Tensor:
        # Mirrors _permuted_layers but shuffles the SEGMENT (M) axis instead.
        n_segments = field.shape[2]
        if self._segment_perm is None or self._segment_perm.shape[0] != n_segments:
            gen = torch.Generator().manual_seed(self.segment_permute_seed)
            self._segment_perm = torch.randperm(n_segments, generator=gen)
        return field[:, :, self._segment_perm]

    def __getitem__(self, i):
        rec = self.records[i]
        images = self._finish(self._load_raw(i))
        return images, float(rec["label"]), self.origin


def zero_channels(field: torch.Tensor, keep: list[str]) -> torch.Tensor:
    # Zero every projection channel NOT in `keep`; channel axis stays width 3.
    mask = torch.tensor(
        [1.0 if p in keep else 0.0 for p in PROJECTIONS], dtype=field.dtype
    ).view(1, 1, 1, N_CHANNELS)
    return field * mask


def collapse_axis_mean(field: torch.Tensor, axis: str) -> torch.Tensor:
    # Average one field axis down to size 1 (kept, not squeezed, so the field
    # stays rank-4). field: (T, L, M, C).
    dim = {"L": 1, "M": 2, "channels": 3}[axis]
    return field.mean(dim=dim, keepdim=True)


def bucket_pool_tokens(field: torch.Tensor, n_buckets: int) -> torch.Tensor:
    # Mean-pool the TOKEN axis (dim 0) into n_buckets contiguous groups, split
    # as evenly as possible. n_buckets >= T is a no-op.
    t = field.shape[0]
    if n_buckets >= t:
        return field
    return torch.stack(
        [chunk.mean(dim=0) for chunk in torch.tensor_split(field, n_buckets, dim=0)],
        dim=0,
    )


def normalize(field: torch.Tensor, stats: dict, n_channels: int = N_CHANNELS) -> torch.Tensor:
    # Standardise per channel (Q, K, V separately) -- they have different
    # magnitudes under GQA, so a single global stat would let one dominate.
    names = PROJECTIONS if n_channels == N_CHANNELS else HIDDEN_STATE_CHANNELS
    mean = torch.tensor(
        [stats[p]["mean"] for p in names], dtype=field.dtype
    ).view(1, 1, 1, n_channels)
    std = torch.tensor(
        [stats[p]["std"] for p in names], dtype=field.dtype
    ).view(1, 1, 1, n_channels)
    return (field - mean) / std.clamp(min=1e-6)


def _set_stats(dataset, stats) -> None:
    dataset.stats = stats


def compute_stats(dataset, indices: list[int], max_examples: int = 500) -> dict:
    # Per-channel mean/std over a sample of the TRAINING split only (val/test
    # here would leak into the model's input scaling).
    # _load_raw is pre-collapse, so stats must use n_channels_on_disk, not n_channels.
    n_channels = getattr(dataset, "n_channels_on_disk", N_CHANNELS)
    names = PROJECTIONS if n_channels == N_CHANNELS else HIDDEN_STATE_CHANNELS

    sums = torch.zeros(n_channels, dtype=torch.float64)
    sqs = torch.zeros(n_channels, dtype=torch.float64)
    count = torch.zeros(n_channels, dtype=torch.float64)

    sample = indices[:max_examples]
    logger.info("computing normalisation stats over %d training examples", len(sample))

    for i in sample:
        x = dataset._load_raw(i).double()         # (T, L, M, n_channels)
        sums += x.sum(dim=(0, 1, 2))
        sqs += (x**2).sum(dim=(0, 1, 2))
        count += x.shape[0] * x.shape[1] * x.shape[2]

    mean = sums / count.clamp(min=1)
    var = (sqs / count.clamp(min=1)) - mean**2
    std = var.clamp(min=0).sqrt()

    return {
        p: {"mean": float(mean[k]), "std": float(max(std[k].item(), 1e-6))}
        for k, p in enumerate(names)
    }


def _pad_stack(images_list: list[torch.Tensor], var_dim: int):
    # Zero-pad to the batch max along var_dim and stack; returns (stacked,
    # mask), mask True at real (non-padded) positions.
    lengths = [img.shape[var_dim] for img in images_list]
    max_len = max(lengths)
    b = len(images_list)

    shape = list(images_list[0].shape)
    shape[var_dim] = max_len
    stacked = torch.zeros(b, *shape, dtype=images_list[0].dtype)
    mask = torch.zeros(b, max_len, dtype=torch.bool)

    for i, img in enumerate(images_list):
        n = img.shape[var_dim]
        idx = [slice(None)] * img.ndim
        idx[var_dim] = slice(0, n)
        stacked[(i, *idx)] = img
        mask[i, :n] = True

    return stacked, mask


def collate(batch):
    # Pad a batch of variable-length responses (T varies per example, axis 0)
    # and build the mask. Returns (images, labels, mask, origins).
    images_list, labels, origins = zip(*batch)
    images, mask = _pad_stack(list(images_list), var_dim=0)
    return (
        images,
        torch.tensor(labels, dtype=torch.float32),
        mask,
        list(origins),
    )
