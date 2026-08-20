"""The detector's CNN backbone: one token feature field -> one embedding vector.

The backbone consumes a 3-channel (Q, K, V) L x M image for a single generated
token and emits an E-dim embedding. The token axis is folded into the batch by
the caller, so the backbone only ever sees a stack of independent images.
"""

from __future__ import annotations

from .resnet18 import IMAGENET_SIZE, ResNet18Adapted
from .scratch_cnn import ResBlock, ScratchCNN

__all__ = [
    "IMAGENET_SIZE",
    "ResBlock",
    "ScratchCNN",
    "ResNet18Adapted",
    "build_backbone",
]


def build_backbone(cfg):
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
    raise ValueError(f"unknown backbone {name!r}")
