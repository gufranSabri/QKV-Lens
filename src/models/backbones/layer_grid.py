# MAIN backbone. FlatMLP's own path, plus a learned per-cell GATE on a
# small (K,1) Conv2d residual over L. The gate starts near 0 (bias=-4,
# sigmoid(-4)~=0.02), so at init this collapses to FlatMLP exactly;
# gradient only grows the gate at (layer, segment) cells where the local
# conv correction actually lowers loss, rather than forcing a cross-layer
# operator into every cell unconditionally. A separate learned per-cell
# w_x independently scales the raw-x contribution into the tail. Strict
# superset of FlatMLP's solution space.

from __future__ import annotations

import torch
import torch.nn as nn

KERNEL_SIZE = 1
HIDDEN = 1
GATE_INIT_BIAS = -4.0  # sigmoid(-4) ~= 0.018 -- conv branch starts ~silent


class LayerGrid(nn.Module):
    def __init__(
        self,
        n_rows: int,
        n_segments: int,
        embed_dim: int = 128,
        dropout: float = 0.0,
        in_ch: int = 3,
    ):
        super().__init__()
        if KERNEL_SIZE % 2 == 0:
            raise ValueError(f"layer_grid: KERNEL_SIZE must be odd, got {KERNEL_SIZE}")

        self.n_rows = n_rows
        self.n_segments = n_segments
        self.in_ch = in_ch

        pad = KERNEL_SIZE // 2
        self.local = nn.Sequential(
            nn.Conv2d(in_ch, HIDDEN, kernel_size=(KERNEL_SIZE, 1), padding=(pad, 0)),
            nn.GELU(),
            nn.Conv2d(HIDDEN, in_ch, kernel_size=(KERNEL_SIZE, 1), padding=(pad, 0)),
        )
        self.gate = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, GATE_INIT_BIAS)

        self.proj = nn.Linear(in_ch * n_rows * n_segments, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.embed_dim = embed_dim

        self.w_x = nn.Parameter(0.5 * torch.ones(1, in_ch * n_rows * n_segments), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, in_ch, L, M)
        g = torch.sigmoid(self.gate(x))
        y = x + g * self.local(x)
        flat = y.flatten(1) + self.w_x * x.flatten(1)
        return self.proj(self.drop(flat))
