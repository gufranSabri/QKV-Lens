# Per-token projection: flatten -> dropout -> Linear; use_proj=False skips the Linear.

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
        use_proj: bool = True,
    ):
        super().__init__()
        in_dim = in_ch * n_rows * n_segments
        self.proj = nn.Linear(in_dim, embed_dim) if use_proj else None
        self.embed_dim = embed_dim if use_proj else in_dim
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, in_ch, L, M)
        x = self.drop(x.flatten(1))            # (N, in_ch*L*M)
        return x if self.proj is None else self.proj(x)  # (N, E)
