"""The detector f_theta: per-token CNN -> temporal encoder -> hallucination logit.

This is QKV-Lens's detector at its paper setting, with the ablation scaffolding
removed. Each token is ONE 3-channel (Q, K, V) image of shape (L, M), so there
is a single CNN stream -- the per-view CNNs and the fusion module that used to
combine them are gone (with three views collapsed onto the channel axis, fusion
was always the identity).

QKV-Steer additionally needs this model to be *differentiable back to its input
field*, because the steering stage takes gradients and Grad-CAM through it. Keep
`encode_tokens` a plain differentiable path -- no torch.no_grad, no detaching.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.dataset import N_CHANNELS
from src.models.backbones import build_backbone
from src.models.temporal import TemporalEncoder


class QKVHalluDetector(nn.Module):
    """Input:  (B, T, 3, L, M) token feature fields + (B, T) padding mask
    Output: (B,) logits -- raw, NOT sigmoided (we use BCEWithLogits).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.backbone = build_backbone(cfg)
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
        """(B, T, 3, L, M) -> (B, T, E) per-token embeddings."""
        b, t, c, h, w = images.shape
        if c != N_CHANNELS:
            raise ValueError(
                f"expected {N_CHANNELS} channels (Q, K, V), got {c}"
            )

        # Fold the token axis into the batch: one CNN call for everything.
        # Never loop over tokens in Python.
        flat = images.reshape(b * t, c, h, w)
        emb = self.backbone(flat)                  # (B*T, E)
        return emb.reshape(b, t, -1)               # (B, T, E)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        combined = self.temporal(self.encode_tokens(images), mask)  # (B, 2H)
        return self.head(combined).squeeze(-1)                      # (B,)


def build_model(cfg) -> QKVHalluDetector:
    return QKVHalluDetector(cfg)
