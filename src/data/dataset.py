"""Dataset + collation for the extracted per-token QKV feature fields."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data import legacy
from src.extract.tensor_ops import PROJECTIONS
from src.utils.logger import get_logger

logger = get_logger(__name__)

#: Channels per token image for the MAIN (QKV) field. Fixed by the paper's
#: feature-field layout. A hidden-states field (see
#: src.extract.tensor_ops.build_hidden_states_field) has 1 channel instead --
#: QKVFieldDataset reads its OWN corpus's actual channel count from
#: geometry.json at construction time (self.n_channels) rather than assuming
#: this module constant; N_CHANNELS itself remains the right default/fallback
#: for the common QKV case and for anything that builds a QKV-shaped tensor
#: directly (e.g. figure scripts that only ever run on QKV corpora).
N_CHANNELS = len(PROJECTIONS)

#: The stats-dict key for a hidden-states field's one channel -- matches
#: geometry["projections"] = ["H"] (see tensor_ops.build_hidden_states_field),
#: so compute_stats/normalize key their dict the same way regardless of which
#: kind of corpus they're given.
HIDDEN_STATE_CHANNELS = ("H",)


class QKVFieldDataset(Dataset):
    """One example = one folder holding tokens.npy of shape (T, L, M, 3).

    Returns (images, label, origin) where images is (T, 3, L, M) -- the channel
    axis is moved into PyTorch's conv position (N, C, H, W) here, so the model
    never has to permute.
    """

    def __init__(
        self,
        root: str | Path,
        stats: dict | None = None,
        max_tokens: int | None = None,
        origin: str | None = None,
        keep_channels: list[str] | None = None,
        token_buckets: int | None = None,
        layer_permute_seed: int | None = None,
        collapse_axis: str | None = None,
    ):
        self.keep_channels = keep_channels
        self.token_buckets = token_buckets
        self.layer_permute_seed = layer_permute_seed
        self.collapse_axis = collapse_axis
        self._layer_perm: torch.Tensor | None = None  # built lazily, see _finish
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
        # Channel count for THIS corpus, read from its own geometry.json
        # rather than assumed to be N_CHANNELS (3): a hidden-states corpus
        # (src.extract.tensor_ops.build_hidden_states_field) writes
        # geometry["projections"] = ["H"], one generic channel, not Q/K/V.
        # Missing/legacy geometry defaults to the QKV case (N_CHANNELS),
        # matching every corpus that predates this field existing at all.
        projections = self.geometry.get("projections")
        # n_channels_on_disk: what's actually stored in tokens.npy, and what
        # `normalize` must standardise over -- per-projection normalisation
        # (see `normalize`'s docstring) has to run BEFORE any channel
        # collapse, or whichever raw projection has the largest magnitude
        # would dominate the average instead of each contributing on a
        # comparable, already-standardised scale. n_channels is the count
        # AFTER collapse_axis='channels' (if set) -- what the MODEL actually
        # sees and what build_model's in_ch must match -- so the two differ
        # exactly when that ablation is active.
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

        # A run-on response truncated at the very first token (the "Q:" turn
        # marker appeared with no real answer before it) leaves n_tokens == 0:
        # no rows in tokens.npy, so an all-False mask reaches the model and
        # temporal.py raises. Drop those here, before any split/stats/loader
        # code sees them, rather than padding around a token axis of length 0.
        empty = [r for r in self.records if r.get("n_tokens", 0) == 0]
        if empty:
            logger.warning(
                "%s: dropping %d example(s) with n_tokens == 0 (empty after "
                "run-on truncation): idx=%s",
                self.root, len(empty), [r["idx"] for r in empty],
            )
            self.records = [r for r in self.records if r.get("n_tokens", 0) > 0]

        # A QKV-Lens-era corpus stores a wider tensor built the same way; it is
        # converted on read rather than re-extracted. Validated ONCE here, at
        # construction, so an incompatible corpus fails before any training
        # starts instead of partway through the first epoch.
        self.legacy = legacy.is_legacy_geometry(self.geometry)
        if self.legacy:
            legacy.assert_compatible(self.geometry, self.root)
            logger.info("%s: %s", self.root, legacy.describe(self.geometry))

        self.stats = stats
        self.max_tokens = max_tokens
        self.origin = origin or f"{self.root.parent.name}_{self.root.name}"

    def __len__(self) -> int:
        return len(self.records)

    @property
    def labels(self) -> list[int]:
        return [int(r["label"]) for r in self.records]

    def _load_raw(self, i) -> torch.Tensor:
        """(T, L, M, 3) for example i, un-normalised. This is the layout the
        normalisation stats are computed and applied over.

        Every tensor enters the pipeline here, so converting a legacy corpus at
        this single point covers `__getitem__` and `compute_stats` alike -- the
        rest of the codebase never sees the QKV-Lens layout.
        """
        rec = self.records[i]
        path = self.root / rec["dir"] / "tokens.npy"

        arr = np.load(path)
        if self.legacy:
            field = legacy.to_field(arr)          # (T, V, L, M, C) -> (T, L, M, 3)
        else:
            field = torch.from_numpy(np.ascontiguousarray(arr)).float()

        if self.max_tokens is not None and field.shape[0] > self.max_tokens:
            field = field[: self.max_tokens]

        if self.token_buckets is not None:
            field = bucket_pool_tokens(field, self.token_buckets)

        return field

    def _finish(self, raw: torch.Tensor) -> torch.Tensor:
        """normalise -> zero unwanted channels -> collapse an axis -> permute layers -> conv position."""
        field = raw
        if self.stats is not None:
            # n_channels_on_disk, not n_channels: standardise over the REAL
            # stored channels (3 for QKV), before any collapse_axis='channels'
            # averaging -- see n_channels_on_disk's docstring above.
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

        # (T, L, M, C) -> (T, C, L, M): channels into conv position.
        return field.permute(0, 3, 1, 2).contiguous()

    def _permuted_layers(self, field: torch.Tensor) -> torch.Tensor:
        """Reorder the LAYER axis (dim 1) with ONE fixed permutation, built on
        first use and reused for every example and every epoch, train and test
        alike -- a per-call random shuffle would let the model see every true
        layer-adjacency relationship anyway (just relabelled per batch), which
        defeats the ablation's whole point (see ModelConfig.layer_permute_seed).
        """
        n_layers = field.shape[1]
        if self._layer_perm is None or self._layer_perm.shape[0] != n_layers:
            gen = torch.Generator().manual_seed(self.layer_permute_seed)
            self._layer_perm = torch.randperm(n_layers, generator=gen)
        return field[:, self._layer_perm]

    def __getitem__(self, i):
        rec = self.records[i]
        images = self._finish(self._load_raw(i))
        return images, float(rec["label"]), self.origin


def zero_channels(field: torch.Tensor, keep: list[str]) -> torch.Tensor:
    """Zero every projection channel NOT in `keep` (Q/K/V ablation).

    field: (T, L, M, 3), channels ordered as PROJECTIONS. The channel AXIS is
    left at width 3 -- only its content is masked -- so the detector's input
    shape, and every (layer, segment, projection) coordinate, is unaffected by
    which projections are actually informative. Applied AFTER normalisation
    (see _finish) so stats are always computed/applied over the full field,
    never skewed by an ablation that only decides what the model gets to see.
    """
    mask = torch.tensor(
        [1.0 if p in keep else 0.0 for p in PROJECTIONS], dtype=field.dtype
    ).view(1, 1, 1, N_CHANNELS)
    return field * mask


def collapse_axis_mean(field: torch.Tensor, axis: str) -> torch.Tensor:
    """Feature-pooling ablation: average ONE field axis down to size 1 --
    "vector aggregation", the compressed alternative the paper's related work
    argues the structured field avoids (see ModelConfig.collapse_axis).

    field: (T, L, M, C). Applied AFTER normalisation and keep_channels (see
    _finish) so this ablation only decides what shape reaches the backbone,
    never skews the statistics the rest of the field is standardised against.

    Returns (T, 1, M, C) for axis="L", (T, L, 1, C) for axis="M", or
    (T, L, M, 1) for axis="channels" -- the axis is kept at size 1, not
    squeezed away, so the field stays rank-4 and every OTHER caller (the
    (T, C, L, M) permute right after this, the backbone's field_shape/in_ch)
    sees an ordinary (if degenerate) field rather than a special case.
    """
    dim = {"L": 1, "M": 2, "channels": 3}[axis]
    return field.mean(dim=dim, keepdim=True)


def bucket_pool_tokens(field: torch.Tensor, n_buckets: int) -> torch.Tensor:
    """Mean-pool the TOKEN axis (dim 0) down to `n_buckets` contiguous groups.

    field: (T, L, M, 3). Token-axis-compression ablation: does the detector
    need one embedding per generated token, or does aggregating nearby tokens
    lose the signal? `n_buckets >= T` is a no-op (every token keeps its own
    bucket).

    Buckets split T as evenly as possible (sizes differ by at most 1, matching
    `torch.tensor_split`) rather than requiring T % n_buckets == 0, since
    response length T varies per example.
    """
    t = field.shape[0]
    if n_buckets >= t:
        return field
    return torch.stack(
        [chunk.mean(dim=0) for chunk in torch.tensor_split(field, n_buckets, dim=0)],
        dim=0,
    )


def normalize(field: torch.Tensor, stats: dict, n_channels: int = N_CHANNELS) -> torch.Tensor:
    """Standardise PER CHANNEL (Q, K, V separately for a QKV field; the one
    channel's own stats for a hidden-states field).

    A single global statistic would be wrong: Q, K and V have very different
    magnitudes -- under GQA they are pooled from different-width vectors -- so
    whichever projection happens to have the largest scale would dominate the
    shared conv filters before the detector ever got a say. (A hidden-states
    field has only one channel, so this degenerates to ordinary
    standardisation there -- the per-channel machinery still applies, just
    with one channel instead of three.)

    field: (T, L, M, n_channels). `stats` is keyed by channel name
    (PROJECTIONS for QKV, or HIDDEN_STATE_CHANNELS for hidden states) --
    whichever `compute_stats` was given when it built this dict.
    """
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
    """Per-channel mean/std over a sample of the TRAINING split only.

    Computed on train indices exclusively -- using val/test examples here would
    leak their distribution into the model's input scaling. Reads the
    dataset's OWN channel count/names (set at construction from its
    geometry.json), so this works unmodified for both a QKV corpus (3 named
    channels) and a hidden-states corpus (1).
    """
    n_channels = getattr(dataset, "n_channels", N_CHANNELS)
    names = PROJECTIONS if n_channels == N_CHANNELS else HIDDEN_STATE_CHANNELS

    # Welford would be tidier, but a two-pass over a bounded sample is simpler
    # and plenty accurate for a normalisation constant.
    sums = torch.zeros(n_channels, dtype=torch.float64)
    sqs = torch.zeros(n_channels, dtype=torch.float64)
    count = torch.zeros(n_channels, dtype=torch.float64)

    sample = indices[:max_examples]
    logger.info("computing normalisation stats over %d training examples", len(sample))

    # Read RAW values via _load_raw -- (T, L, M, n_channels), before normalisation.
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
    """Zero-pad a list of tensors to the batch max along `var_dim` and stack.

    Returns (stacked, mask) where mask is (B, max_len) bool, True at real
    (non-padded) positions along `var_dim`.
    """
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
    """Pad a batch of variable-length responses and build the mask.

    Images are (T, 3, L, M) -- T (generated tokens) varies per example and is
    padded/masked on axis 0.

    Returns:
        images: (B, T_max, 3, L, M)
        labels: (B,)
        mask:   (B, T_max) bool -- True at real tokens
        origins: list[str]
    """
    images_list, labels, origins = zip(*batch)
    images, mask = _pad_stack(list(images_list), var_dim=0)
    return (
        images,
        torch.tensor(labels, dtype=torch.float32),
        mask,
        list(origins),
    )
