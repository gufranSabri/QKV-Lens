from __future__ import annotations

from .flat_mlp import FlatMLP
from .layer_grid import LayerGrid

__all__ = [
    "LayerGrid",
    "FlatMLP",
    "build_backbone",
]

# Backbones needing field_shape up front (shape-dependent first/last layer).
_NEEDS_FIELD_SHAPE = ("flat_mlp", "layer_grid")


def build_backbone(cfg, field_shape: tuple[int, int] | None = None, in_ch: int | None = None):
    if in_ch is None:
        from src.data.dataset import N_CHANNELS
        in_ch = N_CHANNELS

    name = cfg.model.backbone

    if name in _NEEDS_FIELD_SHAPE and field_shape is None:
        raise ValueError(
            f"model.backbone={name!r} needs field_shape=(n_rows, n_segments) "
            "-- see build_model's own field_shape argument."
        )

    if name == "flat_mlp":
        n_rows, n_segments = field_shape
        return FlatMLP(
            n_rows=n_rows, n_segments=n_segments,
            embed_dim=cfg.model.embed_dim, dropout=cfg.model.dropout, in_ch=in_ch,
        )
    if name == "layer_grid":
        n_rows, n_segments = field_shape
        return LayerGrid(
            n_rows=n_rows, n_segments=n_segments,
            embed_dim=cfg.model.embed_dim, dropout=cfg.model.dropout, in_ch=in_ch,
            use_gate=cfg.model.layer_grid_use_gate,
            use_conv=cfg.model.layer_grid_use_conv,
            use_skip=cfg.model.layer_grid_use_skip,
        )
    raise ValueError(f"unknown backbone {name!r}")
