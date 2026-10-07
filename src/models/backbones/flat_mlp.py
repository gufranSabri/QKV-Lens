# Structure-preservation ablation baseline: no spatial stage at all, just
# flatten -> dropout -> Linear. Shares LayerCNN's tail exactly.

from __future__ import annotations

import torch
import torch.nn as nn


class FlatMLP(nn.Module):
    def __init__(
        self,
        n_rows: int,
        n_segments: int,
        embed_dim: int = 128,
        dropout: float = 0.0,
        in_ch: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        in_dim = in_ch * n_rows * n_segments
        self.proj = nn.Linear(in_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, in_ch, L, M)
        x = x.flatten(1)                       # (N, in_ch*L*M)
        return self.proj(self.drop(x))         # (N, E)
