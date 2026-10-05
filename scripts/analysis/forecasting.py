"""How early in a response can the detector call a hallucination?

Runs a trained detector over every PREFIX of every held-out response: at step t
the detector sees tokens 1..t and nothing after. Sweeping t from 1 to T gives one
probability trajectory per example, and the aggregate answers "how much of the
response do we need before the prediction is right and stays right".

WHY THIS EXISTS (QKV-Steer)
---------------------------
Steering needs an attribution map computed DURING generation, not after it. The
plan's closed-loop drift mode (§1.5c) refreshes the map every k tokens from the
prefix generated so far. That is only worth doing if the detector is informative
on a prefix at all -- if it needs the full response, closed-loop steering has
nothing to act on early, and the intervention can only ever be late. This script
measures that directly, before any steering code is written.

PREFIXES ARE SLICED, NOT MASKED
-------------------------------
A prefix is `images[:, :t]`, not the full tensor with a shortened mask. The two
are NOT equivalent: TemporalEncoder zeroes padding before its Conv1d, but a
kernel_size=3 conv at position t-1 still reads position t, so a masked "prefix"
gives its boundary token a receptive field that includes zeroed padding, while a
sliced prefix ends there for real. Slicing is what the detector would genuinely
see mid-generation, so slicing is what this measures. (Verified: the two agree
only at t == T.)

Batching therefore loops OUTER over prefix length and batches examples INNER --
every row in a batch shares one t, which is exact (verified against per-example
calls), whereas mixing lengths in a batch would require the masking that is
precisely what we are avoiding.
"""

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
from src.utils.metrics import compute_metrics
from src.utils.progress import progress
from src.utils.seed import pick_device, seed_everything

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
    """One example's per-prefix detector output."""

    idx: int                  # example index in the corpus
    label: int                # 1 = hallucinated
    probs: np.ndarray         # (T,) p(hallucinated) after seeing tokens 1..t

    @property
    def n_tokens(self) -> int:
        return len(self.probs)

    @property
    def correct(self) -> np.ndarray:
        """(T,) bool -- was the thresholded call right after t tokens?"""
        pred = (self.probs >= THRESHOLD).astype(int)
        return pred == self.label

    def first_stable_correct(self) -> int | None:
        """Earliest t (1-indexed) after which the call is correct and STAYS
        correct through the end of the response.

        Returns None if the final call is wrong -- there is no such t, and
        counting a transient early hit would overstate how early detection
        happens. This is the statistic the histogram reports.
        """
        c = self.correct
        if not c[-1]:
            return None
        # Walk back from the end while the run of correctness is unbroken.
        t = len(c)
        while t > 1 and c[t - 2]:
            t -= 1
        return t

    def lockin_fraction(self) -> float:
        """`first_stable_correct` as a fraction of the response, or NaN.

        NaN (not 1.0) when the final call is wrong: "never locks in" is a
        different outcome from "locks in only at the very end", and averaging
        the two together would report a detector that fails on an example as
        though it succeeded late. Every consumer filters NaN and reports the
        never-count alongside the median.
        """
        t = self.first_stable_correct()
        return float("nan") if t is None else t / self.n_tokens

    def fraction_of(self, values: np.ndarray) -> np.ndarray:
        """Resample a per-token array onto FRACTIONS via nearest token.

        Nearest-token (not interpolation): the underlying quantity at a given
        prefix is a real measurement at an integer t, and averaging two adjacent
        prefixes would invent a prediction the detector never made.
        """
        # ceil so fraction f maps to at least 1 token and f=1.0 maps to T.
        idx = np.ceil(FRACTIONS * self.n_tokens).astype(int)
        idx = np.clip(idx, 1, self.n_tokens) - 1
        return values[idx]


# ---------------------------------------------------------------------------
# Cache
#
# The sweep is T detector passes per example and needs the extracted field plus
# a GPU; redrawing a figure needs neither. Every trajectory is therefore stored
# verbatim -- ragged, since responses differ in length -- so the figures and the
# pooled summary rebuild from cache in seconds. Storing only the 13-point
# FRACTIONS resample would have been smaller, but lock-in is a per-token
# statistic: it cannot be recovered from a resampled curve.
# ---------------------------------------------------------------------------


def save_trajectories(trajs: list[Trajectory], dest: Path, provenance: dict) -> Path:
    """Write trajectories to `dest` (.npz), ragged, with their provenance."""
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
    """Inverse of `save_trajectories`."""
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
    """p(hallucinated) at every prefix length, for a group of equal-length examples.

    `fields` are (T, 3, L, M) tensors that all share the same T. Returns one
    (T,) array per input. Loops outer over t so every batched call uses a single
    prefix length -- see the module docstring for why mixing lengths is unsafe.
    """
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
    """Run the detector over every prefix of every evaluated example."""
    from src.train import load_source

    device = pick_device()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    source = load_source(cfg, dataset_name, cfg.llm.alias)
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

    model = build_model(cfg).to(device)
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
