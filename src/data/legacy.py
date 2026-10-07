# Reads QKV-Lens-era extracted corpora without re-extracting them.
#
# QKV-Lens wrote (T, V, L, M, C) where V is the Q/K/V view axis and channel 0
# is the raw mean-pooled activation -- the same quantity QKV-Steer's (T, L, M,
# 3) field is defined in terms of. Conversion is field = old[..., 0] with V
# moved last: exact, not approximate. Only valid when the legacy corpus was
# mean-pooled (QKV-Steer fixes pooling to mean); anything else is rejected,
# not reinterpreted. boundary_mode is not checked -- it only affected the
# delta channels, which this conversion discards.

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)

LEGACY_VIEWS = ("Q", "K", "V")
REQUIRED_POOL = "mean"
RAW_CHANNEL = 0

# QKV-Lens tree layout: {data_root}/{source}/{extraction_type}/{dataset}/{llm_alias}/
# Only "qkv" source is listed -- "hs" has no Q/K/V structure to steer.
LEGACY_PREFIXES = (("qkv", "transforms"), ("qkv", "delta"))


def resolve_root(
    native: Path, data_root: Path, dataset: str, llm_alias: str, is_default_pool: bool = True
) -> Path:
    # native: where a fresh extraction under the caller's config would write.
    # is_default_pool=False (a pooling-ablation config) skips the legacy
    # fallback entirely -- a legacy tree is always mean-pooled, and falling
    # back to it for a non-default pool request would silently swap in data
    # the caller didn't ask for. A fresh QKV-Steer corpus always wins.
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

    return native


def is_legacy_geometry(geometry: dict) -> bool:
    # Keyed on "views", which QKV-Steer never writes and QKV-Lens always did.
    return "views" in geometry


def assert_compatible(geometry: dict, root: Path) -> None:
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
    return (
        f"QKV-Lens corpus (source={geometry.get('source')}, "
        f"extraction_type={geometry.get('extraction_type')}, "
        f"pool={geometry.get('pool')}, views={geometry.get('views')}) "
        f"-> converting channel {RAW_CHANNEL} (raw mean-pooled) to the "
        f"(T, L, M, 3) QKV-Steer field"
    )


def to_field(arr: np.ndarray) -> torch.Tensor:
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
