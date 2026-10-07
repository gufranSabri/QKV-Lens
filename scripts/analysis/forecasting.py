from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.config import Config
from src.data.dataset import normalize
from src.models.classifier import build_model
from src.utils.logger import get_logger
from src.utils.progress import progress
from src.utils.seed import pick_device

logger = get_logger(__name__)

#: Prefix fractions the pooled figures report on. Responses have different
#: lengths, so an absolute token index is not comparable across examples; a
#: fraction is. 0.05 -> "the first 5% of the response".
FRACTIONS = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                      0.60, 0.70, 0.80, 0.90, 1.00])

#: Decision threshold on p(hallucinated). 0.5 matches src.utils.metrics.
THRESHOLD = 0.5


@dataclass
class Trajectory:
    idx: int
    label: int                # 1 = hallucinated
    probs: np.ndarray         # (T,) p(hallucinated) after seeing tokens 1..t

    @property
    def n_tokens(self) -> int:
        return len(self.probs)

    @property
    def correct(self) -> np.ndarray:
        pred = (self.probs >= THRESHOLD).astype(int)
        return pred == self.label

    def first_stable_correct(self) -> int | None:
        # Earliest t after which the call is correct and stays correct to the
        # end; None if the final call is wrong (no such t).
        c = self.correct
        if not c[-1]:
            return None
        t = len(c)
        while t > 1 and c[t - 2]:
            t -= 1
        return t

    def lockin_fraction(self) -> float:
        # NaN (not 1.0) when the final call is wrong, so "never locks in" is
        # distinguishable from "locks in at the very end".
        t = self.first_stable_correct()
        return float("nan") if t is None else t / self.n_tokens

    def fraction_of(self, values: np.ndarray) -> np.ndarray:
        # Nearest-token, not interpolated: each prefix is a real measurement.
        idx = np.ceil(FRACTIONS * self.n_tokens).astype(int)
        idx = np.clip(idx, 1, self.n_tokens) - 1
        return values[idx]


def save_trajectories(trajs: list[Trajectory], dest: Path, provenance: dict) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    lengths = np.array([t.n_tokens for t in trajs], dtype=np.int64)
    np.savez_compressed(
        dest,
        idx=np.array([t.idx for t in trajs], dtype=np.int64),
        label=np.array([t.label for t in trajs], dtype=np.int64),
        lengths=lengths,
        # Ragged: one flat buffer plus the offsets that cut it back apart.
        probs_concat=np.concatenate([t.probs for t in trajs]).astype(np.float32),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        provenance=np.array(json.dumps(provenance)),
    )
    return dest


def load_trajectories(src: Path) -> tuple[list[Trajectory], dict]:
    with np.load(src, allow_pickle=False) as z:
        offsets, probs = z["offsets"], z["probs_concat"]
        trajs = [
            Trajectory(
                idx=int(z["idx"][i]),
                label=int(z["label"][i]),
                probs=probs[offsets[i]:offsets[i + 1]].astype(np.float64),
            )
            for i in range(len(z["idx"]))
        ]
        provenance = json.loads(str(z["provenance"]))
    return trajs, provenance


@torch.no_grad()
def _prefix_probs(
    model, fields: list[torch.Tensor], device, batch_size: int
) -> list[np.ndarray]:
    # Loops outer over prefix length t, sliced (not masked) -- a masked prefix
    # would let the Conv1d's receptive field leak into zeroed padding.
    n = len(fields)
    if n == 0:
        return []
    n_tokens = fields[0].shape[0]
    stacked = torch.stack(fields).to(device)          # (n, T, 3, L, M)
    out = np.zeros((n, n_tokens), dtype=np.float64)

    for t in range(1, n_tokens + 1):
        for lo in range(0, n, batch_size):
            chunk = stacked[lo : lo + batch_size, :t]         # (b, t, 3, L, M)
            mask = torch.ones(chunk.shape[0], t, dtype=torch.bool, device=device)
            logits = model(chunk, mask)
            out[lo : lo + chunk.shape[0], t - 1] = (
                torch.sigmoid(logits).double().cpu().numpy()
            )

    return [out[i] for i in range(n)]


def compute_trajectories(
    cfg: Config,
    checkpoint: Path,
    dataset_name: str,
    limit: int | None,
    batch_size: int,
) -> list[Trajectory]:
    from src.train import load_source

    device = pick_device()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    source = load_source(cfg, dataset_name, cfg.llm.alias, field_source=cfg.extract.source)
    source.stats = ckpt["stats"]

    # Held-out rows only. Training rows would make early detection look better
    # than it is -- the detector has already fit them.
    heldout = ckpt.get("heldout_idx")
    if not heldout:
        raise ValueError(
            f"{checkpoint} carries no heldout_idx; it predates the held-out "
            "split or was trained on another corpus. Retrain before forecasting."
        )
    indices = list(heldout)
    if ckpt.get("llm_alias") not in (None, cfg.llm.alias):
        raise ValueError(
            f"checkpoint was trained on llm={ckpt['llm_alias']!r} but the config "
            f"asks for {cfg.llm.alias!r}; heldout_idx does not transfer across LLMs."
        )
    if limit is not None:
        indices = indices[:limit]

    model = build_model(
        cfg, field_shape=ckpt.get("field_shape"), in_ch=ckpt.get("in_ch")
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Group by token count so each _prefix_probs call gets equal-length inputs.
    by_len: dict[int, list[int]] = {}
    for i in indices:
        by_len.setdefault(int(source.records[i]["n_tokens"]), []).append(i)

    logger.info(
        "forecasting over %d held-out examples (%d distinct lengths), "
        "%d total detector passes",
        len(indices), len(by_len),
        sum(len(v) * k for k, v in by_len.items()),
    )

    trajectories: list[Trajectory] = []
    for n_tokens, group in progress(
        sorted(by_len.items()), desc="prefix sweep", ncols=100
    ):
        fields, labels = [], []
        for i in group:
            raw = source._load_raw(i)                      # (T, L, M, 3)
            field = normalize(raw, source.stats).permute(0, 3, 1, 2).contiguous()
            fields.append(field)
            labels.append(int(source.records[i]["label"]))

        # A stored tensor can be shorter than its manifest n_tokens once
        # extract.max_tokens truncates, so trust the tensor.
        actual = {f.shape[0] for f in fields}
        if len(actual) != 1:
            raise RuntimeError(
                f"length group {n_tokens} holds mixed tensor lengths {sorted(actual)}"
            )

        for i, lab, probs in zip(
            group, labels, _prefix_probs(model, fields, device, batch_size)
        ):
            trajectories.append(Trajectory(idx=i, label=lab, probs=probs))

    return trajectories
