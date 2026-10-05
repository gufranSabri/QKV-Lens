"""Read QKV-Lens-era extracted corpora without re-extracting them.

QKV-Lens wrote a wider tensor than QKV-Steer needs:

    QKV-Lens   (T, V, L, M, C)   V = view axis (Q, K, V)
                                 C = extraction channels, whose meaning depends
                                     on extraction_type:
                                       delta      (raw, delta-prev, delta-next)
                                       transforms (raw, DWT-Haar, DWT-Sym3)
    QKV-Steer  (T, L, M, 3)      trailing axis = the projections (Q, K, V)

The paper's Algorithm 1 field is recoverable from the QKV-Lens tensor exactly,
with no re-extraction and no approximation, because QKV-Lens's channel 0 IS the
mean-pooled activation -- the same quantity Algorithm 1 defines. The other two
channels are derived quantities (layer deltas or wavelet magnitudes) that
QKV-Steer does not use, so dropping them loses nothing the method needs:

    field = old[..., 0].transpose(V -> last)

i.e. take the raw channel of each view, then move the view axis to the end so it
becomes the (Q, K, V) projection axis.

WHEN THIS IS VALID
------------------
Only when the legacy corpus was pooled the way the paper specifies. QKV-Lens
made pooling configurable (max/mean/l2/sdk); QKV-Steer fixes it to mean, because
a steering coordinate has to refer to the same quantity the detector saw. A
corpus pooled any other way is NOT the paper's field and is rejected rather than
silently reinterpreted -- see `assert_compatible`.

`boundary_mode` is deliberately NOT checked: it only ever affected the delta
channels, which this conversion discards.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)

#: The view order QKV-Lens stored on its view axis (run_extraction.VIEWS).
#: This is what makes the converted channel axis mean (Q, K, V) in that order.
LEGACY_VIEWS = ("Q", "K", "V")

#: The pooling QKV-Steer's feature field is defined in terms of. A legacy corpus
#: pooled differently is not the paper's field (QKV-Lens Table 3 ablated these).
REQUIRED_POOL = "mean"

#: Index of the raw mean-pooled channel on the legacy channel axis. Channel 0 is
#: `pooled` under BOTH extraction_types (see the old tensor_ops.add_delta_channels
#: and add_transform_channels, which each stack `pooled` first).
RAW_CHANNEL = 0


#: Where a QKV-Lens tree nests a corpus, relative to data_root:
#:     {data_root}/{source}/{extraction_type}/{dataset}/{llm_alias}/
#: QKV-Steer writes {data_root}/{dataset}/{llm_alias}/ with no such prefix.
#: Only the `qkv` source is listed -- `hs` has no Q/K/V structure to steer, so
#: it is never a fallback candidate. `transforms` is preferred over `delta`
#: purely for determinism; channel 0 is the same mean-pooled activation in both.
LEGACY_PREFIXES = (("qkv", "transforms"), ("qkv", "delta"))


def resolve_root(
    native: Path, data_root: Path, dataset: str, llm_alias: str, is_default_pool: bool = True
) -> Path:
    """Locate a corpus, preferring the QKV-Steer layout and falling back to a
    QKV-Lens tree.

    Args:
        native: the path a fresh extraction under the CALLER's config would
            write to -- i.e. `cfg.example_dir()`, already carrying any
            non-default `extract.pool` prefix (see Config.example_dir).
        is_default_pool: False for a pooling-ablation config (extract.pool !=
            "mean"). A legacy QKV-Lens tree is a mean-pooled corpus in
            disguise (see `assert_compatible`'s `pool` check) -- falling back
            to it for a NON-default pool request would silently hand the
            caller mean-pooled data under a config that asked for `max` or
            `strided`, so the fallback is skipped entirely in that case.

    Returns the QKV-Steer path unchanged when it exists (or when no legacy tree
    applies), so a fresh extraction always wins and the fallback can never
    shadow it. This keeps `data_root` a single setting rather than forcing the
    user to spell out the old {source}/{extraction_type} levels.
    """
    if (native / "manifest.jsonl").exists():
        return native

    if is_default_pool:
        for prefix in LEGACY_PREFIXES:
            candidate = data_root.joinpath(*prefix, dataset, llm_alias)
            if (candidate / "manifest.jsonl").exists():
                logger.info(
                    "no QKV-Steer corpus at %s; using the QKV-Lens tree at %s",
                    native, candidate,
                )
                return candidate

    # Nothing found: return the native path so the caller's own "run extract
    # first" error names the location a fresh extraction would write to.
    return native


def is_legacy_geometry(geometry: dict) -> bool:
    """True if this geometry.json was written by QKV-Lens.

    Keyed on `views`, which QKV-Steer never writes and QKV-Lens always did.
    Falling back to tensor rank would be fragile; the geometry file is the
    authoritative record of how a corpus was built.
    """
    return "views" in geometry


def assert_compatible(geometry: dict, root: Path) -> None:
    """Reject a legacy corpus that is not the paper's feature field.

    Raises with an actionable message rather than converting something whose
    channel 0 is not the mean-pooled activation Algorithm 1 defines.
    """
    pool = geometry.get("pool")
    if pool != REQUIRED_POOL:
        raise ValueError(
            f"{root} is a QKV-Lens corpus pooled with pool={pool!r}, but "
            f"QKV-Steer's feature field is defined with {REQUIRED_POOL!r} "
            "pooling (QKV-Lens Table 3, and the paper's Algorithm 1 Eq. 4).\n"
            "Its channel 0 is therefore not the quantity a steering coordinate "
            "refers to. Re-extract this (dataset, LLM) with QKV-Steer instead."
        )

    views = geometry.get("views")
    if list(views) != list(LEGACY_VIEWS):
        raise ValueError(
            f"{root} stores views={views}, but the QKV-Steer field's channel "
            f"axis is exactly {list(LEGACY_VIEWS)} in that order. A corpus "
            "extracted with a view subset cannot be widened back; re-extract it."
        )

    source = geometry.get("source")
    if source != "qkv":
        raise ValueError(
            f"{root} was extracted with source={source!r} (hidden states), which "
            "has no Q/K/V structure to steer. QKV-Steer reads the qkv corpus; "
            "point data_root at the qkv tree, or re-extract."
        )


def describe(geometry: dict) -> str:
    """One-line summary of what a legacy corpus is, for the run log."""
    return (
        f"QKV-Lens corpus (source={geometry.get('source')}, "
        f"extraction_type={geometry.get('extraction_type')}, "
        f"pool={geometry.get('pool')}, views={geometry.get('views')}) "
        f"-> converting channel {RAW_CHANNEL} (raw mean-pooled) to the "
        f"(T, L, M, 3) QKV-Steer field"
    )


def to_field(arr: np.ndarray) -> torch.Tensor:
    """QKV-Lens (T, V, L, M, C) -> QKV-Steer (T, L, M, 3) float tensor.

    Keeps only the raw channel of each view, then moves the view axis last so it
    becomes the projection axis. Exact, not approximate: the values are the same
    mean-pooled numbers QKV-Lens stored.
    """
    if arr.ndim != 5:
        raise ValueError(
            f"expected a legacy (T, V, L, M, C) tensor, got shape {arr.shape}"
        )
    n_views = arr.shape[1]
    if n_views != len(LEGACY_VIEWS):
        raise ValueError(
            f"legacy tensor has {n_views} views, expected {len(LEGACY_VIEWS)} "
            f"({', '.join(LEGACY_VIEWS)})"
        )

    raw = arr[..., RAW_CHANNEL]                    # (T, V, L, M)
    field = np.transpose(raw, (0, 2, 3, 1))        # (T, L, M, V) -- V = Q,K,V
    return torch.from_numpy(np.ascontiguousarray(field)).float()
