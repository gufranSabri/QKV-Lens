"""FlatMLP: the flattened-QKV structure-preservation ablation baseline.

QKV-Lens's central claim is that preserving the (L, M) organization -- not
merely having access to the Q/K/V values -- is what ScratchCNN exploits. This
backbone is the control: EVERY stage of ScratchCNN (stem, ResBlocks, pool)
that actually operates over the (L, M) spatial layout is removed outright,
leaving only what comes after it unchanged -- dropout, then one Linear to
embed_dim. The field is flattened before that Linear ever sees it, so its
input is the same (Q, K, V) values ScratchCNN gets, with every spatial
relation between them destroyed. If ScratchCNN beats this, the structure
itself -- not just the values -- is doing work.

Deliberately NOT capacity-matched to ScratchCNN via an extra hidden layer: the
two backbones' architectures are otherwise identical (same dropout, same
single Linear to embed_dim), isolating the ablation to exactly one thing --
the presence of the spatial stage -- rather than trading one confound
(unequal depth) for another (an extra learned layer that only one baseline
gets). The one honest asymmetry this leaves: `proj`'s fan-in is in_ch*L*M
(e.g. 3072) here vs. ScratchCNN's fixed 128, so this backbone has MORE raw
parameters in that single Linear -- a side effect of comparing "a CNN that
pools down to 128 before its head" against "no pooling at all", not a knob
tuned to equalize parameter count.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FlatMLP(nn.Module):
    """Flatten (in_ch, L, M) -> one vector -> dropout -> Linear -> (N, embed_dim).

    Mirrors ScratchCNN's forward exactly from `self.drop` onward (`return
    self.proj(self.drop(x))`); only what feeds `x` differs -- ScratchCNN's
    CNN-pooled 128-dim vector vs. this backbone's flattened in_ch*L*M vector.
    """

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
