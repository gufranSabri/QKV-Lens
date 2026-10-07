# Snapshot of layer_cnn.py from before the no-upsample 3-conv rewrite --
# single Conv2d + nn.Upsample resize + separate channel_proj. Not wired into
# build_backbone -- reference only.

from __future__ import annotations

import torch
import torch.nn as nn

KERNEL_SIZE: int | None = 5
STRIDE: int | None = 5
CONV_OUT_CHANNELS = 3


def _checked_padding(length: int, kernel: int, stride: int) -> int:
    if kernel > length + 2 * (kernel - 1):
        raise ValueError(
            f"layer_cnn: kernel_size={kernel} cannot fit inside the "
            f"layer axis (length {length}) even with padding."
        )
    remainder = (length - kernel) % stride
    if remainder == 0:
        return 0
    pad_total = stride - remainder
    if pad_total % 2 != 0:
        pad_total += 1
    return pad_total // 2


class LayerCNN(nn.Module):
    def __init__(
        self,
        n_rows: int,
        n_segments: int,
        embed_dim: int = 128,
        dropout: float = 0.0,
        in_ch: int = 3,
    ):
        super().__init__()
        self.n_rows = n_rows
        self.n_segments = n_segments
        self.in_ch = in_ch

        kernel = KERNEL_SIZE if KERNEL_SIZE is not None else n_rows
        stride = STRIDE if STRIDE is not None else kernel
        padding = _checked_padding(n_rows, kernel, stride)
        self.conv = nn.Conv2d(
            in_ch, CONV_OUT_CHANNELS,
            kernel_size=(kernel, 1), stride=(stride, 1), padding=(padding, 0),
        )
        self.act = nn.GELU()
        l_out = (n_rows + 2 * padding - kernel) // stride + 1
        self.resize = (
            nn.Identity()
            if l_out == n_rows
            else nn.Upsample(size=(n_rows, n_segments), mode="bilinear", align_corners=False)
        )
        self.channel_proj = nn.Conv2d(CONV_OUT_CHANNELS, in_ch, kernel_size=1)

        self.proj = nn.Linear(in_ch * n_rows * n_segments, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv(x))
        x = self.resize(x)
        x = self.channel_proj(x)
        flat = x.flatten(1)
        return self.proj(self.drop(flat))
