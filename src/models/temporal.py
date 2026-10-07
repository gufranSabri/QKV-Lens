# Temporal head: token embeddings -> one hallucination logit. Every op here
# must respect the padding mask, or padding silently corrupts the pooled
# representation and every downstream metric.

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D), mask: (B, T) bool -- True = real token
        logits = self.score(x).squeeze(-1)                    # (B, T)
        logits = logits.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = (weights * x).sum(dim=1)                     # (B, D)
        if return_weights:
            return pooled, weights.squeeze(-1)                # (B, D), (B, T)
        return pooled


class TemporalEncoder(nn.Module):
    # Conv1d (local n-grams) -> BiLSTM (long range) -> masked attention pool.

    def __init__(
        self,
        input_dim: int,
        conv_layers: int = 2,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        convs = []
        for _ in range(conv_layers):
            convs += [
                nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(input_dim),
                nn.GELU(),
            ]
        self.convs = nn.Sequential(*convs) if convs else None

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.pool = MaskedAttentionPool(2 * lstm_hidden)
        self.drop = nn.Dropout(dropout)
        self.out_dim = 2 * lstm_hidden

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F), mask: (B, T) bool
        if not mask.any(dim=1).all():
            empty = (~mask.any(dim=1)).nonzero().flatten().tolist()
            raise ValueError(
                f"batch rows {empty} have an all-False mask (zero real tokens). "
                "Every example must have at least one token."
            )

        if self.convs is not None:
            # zero padding before convolving so it can't bleed into real tokens
            x = x * mask.unsqueeze(-1)
            x = self.convs(x.transpose(1, 2)).transpose(1, 2)   # (B, T, F)
            x = x * mask.unsqueeze(-1)

        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=mask.shape[1]
        )

        return self.pool(self.drop(out), mask, return_weights=return_weights)
