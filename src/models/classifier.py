"""The detector f_theta: per-token backbone -> temporal encoder -> hallucination logit.

This is QKV-Lens's detector at its paper setting, with the ablation scaffolding
removed. Each token is ONE 3-channel (Q, K, V) image of shape (L, M), so there
is a single backbone stream -- the per-view CNNs and the fusion module that
used to combine them are gone (with three views collapsed onto the channel
axis, fusion was always the identity). The backbone itself is a config choice
(see src/models/backbones/) -- flat_mlp (no spatial structure) is the main
approach as of the structure-preservation ablation; scratch_cnn remains as
the "with spatial structure" comparison arm.

QKV-Steer additionally needs this model to be *differentiable back to its input
field*, because the steering stage takes Integrated Gradients through it (see
src/cam.py). Keep `encode_tokens` a plain differentiable path -- no
torch.no_grad, no detaching.
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
        """(B, T, C, L, M) -> (B, T, E) per-token embeddings. C is self.in_ch
        (3 for QKV, 1 for hidden states)."""
        b, t, c, h, w = images.shape
        if c != self.in_ch:
            raise ValueError(f"expected {self.in_ch} channel(s), got {c}")

        # Fold the token axis into the batch: one CNN call for everything.
        # Never loop over tokens in Python.
        flat = images.reshape(b * t, c, h, w)
        emb = self.backbone(flat)                  # (B*T, E)
        return emb.reshape(b, t, -1)               # (B, T, E)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        combined = self.temporal(self.encode_tokens(images), mask)  # (B, 2H)
        return self.head(combined).squeeze(-1)                      # (B,)


def build_model(
    cfg, field_shape: tuple[int, int] | None = None, in_ch: int | None = None
) -> QKVHalluDetector:
    """field_shape = (n_rows, n_segments), i.e. the field's (L, M). Only
    required when cfg.model.backbone needs it to fix its own shape before
    `load_state_dict` can run -- see build_backbone's `_NEEDS_FIELD_SHAPE`.

    in_ch: the field's channel count -- None (default) resolves to N_CHANNELS
    (3, QKV). Pass 1 for a hidden-states corpus. Every call site that loads a
    checkpoint (test.py, cam.py, forecasting.py) must pass the SAME in_ch the
    checkpoint was trained with (saved in the checkpoint as "in_ch", mirroring
    field_shape) -- `ckpt.get("in_ch")` is already None for an older
    checkpoint saved before this field existed, which correctly resolves to
    the QKV default here, exactly what those checkpoints actually are.
    """
    return QKVHalluDetector(cfg, field_shape=field_shape, in_ch=in_ch)
