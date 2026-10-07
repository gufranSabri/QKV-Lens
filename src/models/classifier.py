# Detector f_theta: per-token backbone -> temporal encoder -> hallucination logit.
# encode_tokens must stay a plain differentiable path (no no_grad/detach) --
# src/cam.py takes Integrated Gradients through it.

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.dataset import N_CHANNELS
from src.models.backbones import build_backbone
from src.models.temporal import TemporalEncoder


class QKVHalluDetector(nn.Module):
    # Input:  (B, T, 3, L, M) token feature fields + (B, T) padding mask
    # Output: (B,) logits -- raw, NOT sigmoided (we use BCEWithLogits).

    def __init__(
        self,
        cfg,
        field_shape: tuple[int, int] | None = None,
        in_ch: int | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.in_ch = in_ch if in_ch is not None else N_CHANNELS

        self.backbone = build_backbone(cfg, field_shape=field_shape, in_ch=self.in_ch)
        self.temporal = TemporalEncoder(
            input_dim=cfg.model.embed_dim,
            conv_layers=cfg.model.conv1d_layers,
            lstm_hidden=cfg.model.lstm_hidden,
            lstm_layers=cfg.model.lstm_layers,
            dropout=cfg.model.dropout,
        )

        head_in = self.temporal.out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(head_in, 1),
        )

    def encode_tokens(self, images: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = images.shape
        if c != self.in_ch:
            raise ValueError(f"expected {self.in_ch} channel(s), got {c}")

        flat = images.reshape(b * t, c, h, w)
        emb = self.backbone(flat)                  # (B*T, E)
        return emb.reshape(b, t, -1)               # (B, T, E)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        combined = self.temporal(self.encode_tokens(images), mask)  # (B, 2H)
        return self.head(combined).squeeze(-1)                      # (B,)


def build_model(
    cfg, field_shape: tuple[int, int] | None = None, in_ch: int | None = None
) -> QKVHalluDetector:
    return QKVHalluDetector(cfg, field_shape=field_shape, in_ch=in_ch)
