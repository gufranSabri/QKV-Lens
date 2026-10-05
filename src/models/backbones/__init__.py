"""The detector's CNN backbone: one token feature field -> one embedding vector.

The backbone consumes a 3-channel (Q, K, V) L x M image for a single generated
token and emits an E-dim embedding. The token axis is folded into the batch by
the caller, so the backbone only ever sees a stack of independent images.
"""

from __future__ import annotations

from .flat_mlp import FlatMLP
from .resnet18 import IMAGENET_SIZE, ResNet18Adapted
from .scratch_cnn import ResBlock, ScratchCNN

__all__ = [
    "IMAGENET_SIZE",
    "ResBlock",
    "ScratchCNN",
    "ResNet18Adapted",
    "FlatMLP",
    "build_backbone",
]

#: Backbones whose first layer's shape depends on (L, M), so build_backbone
#: requires `field_shape` for these -- see FlatMLP's docstring for why this
#: can't be solved lazily on first forward (breaks load_state_dict ordering).
_NEEDS_FIELD_SHAPE = ("flat_mlp",)


def build_backbone(cfg, field_shape: tuple[int, int] | None = None):
    from src.data.dataset import N_CHANNELS

    name = cfg.model.backbone

    if name == "scratch_cnn":
        return ScratchCNN(
            embed_dim=cfg.model.embed_dim, dropout=cfg.model.dropout, in_ch=N_CHANNELS
        )
    if name == "resnet18":
        return ResNet18Adapted(
            embed_dim=cfg.model.embed_dim,
            pretrained=cfg.model.pretrained_backbone,
            dropout=cfg.model.dropout,
            in_ch=N_CHANNELS,
        )
    if name == "flat_mlp":
        if field_shape is None:
            raise ValueError(
                "model.backbone='flat_mlp' needs field_shape=(n_rows, n_segments) "
                "-- see build_model's own field_shape argument."
            )
        n_rows, n_segments = field_shape
        return FlatMLP(
            n_rows=n_rows, n_segments=n_segments,
            embed_dim=cfg.model.embed_dim, dropout=cfg.model.dropout, in_ch=N_CHANNELS,
        )
    raise ValueError(f"unknown backbone {name!r}")
