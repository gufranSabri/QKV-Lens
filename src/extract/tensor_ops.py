# Pure tensor math for building per-token QKV feature fields (QKV-Lens
# Algorithm 1), free of any model/HuggingFace dependency.
#
# Q/K/V raw (T, L, D_c) -> mean_pool_segments -> (T, L, M) per channel ->
# stack -> (T, L, M, 3), trailing axis ordered Q, K, V. This is the paper's
# F_{t,l,m,c} (Eq. 4-7) and the coordinate system the steering stage writes
# back into.

from __future__ import annotations

import torch

# Load-bearing order: F[..., c] is addressed by projection name on both the
# read and (future) steering-write paths.
PROJECTIONS = ("Q", "K", "V")

# "mean" is the paper's fixed setting; max/strided exist only to reproduce
# the pooling ablation (QKV-Lens Table 3), never for steering.
POOL_MODES = ("mean", "max", "strided")


def mean_pool_segments(raw: torch.Tensor, n_segments: int) -> torch.Tensor:
    return pool_segments(raw, n_segments, mode="mean")


def pool_segments(raw: torch.Tensor, n_segments: int, mode: str = "mean") -> torch.Tensor:
    # raw: (..., D) -> (..., n_segments). D must be divisible by n_segments --
    # a ragged final segment would cover a different span than the others.
    # mean/max pool each contiguous S-wide chunk; strided takes raw[..., ::S].
    if mode not in POOL_MODES:
        raise ValueError(f"pool mode must be one of {POOL_MODES}, got {mode!r}")

    d = raw.shape[-1]
    if n_segments <= 0:
        raise ValueError(f"n_segments must be positive, got {n_segments}")
    if d % n_segments != 0:
        raise ValueError(
            f"feature dim D={d} is not divisible by n_segments={n_segments}; "
            "pooling would drop or duplicate dimensions"
        )

    segment_size = d // n_segments

    if mode == "strided":
        return raw[..., ::segment_size][..., :n_segments]

    chunked = raw.reshape(*raw.shape[:-1], n_segments, segment_size)
    if mode == "max":
        return chunked.amax(dim=-1)
    return chunked.mean(dim=-1)


def build_feature_field(
    activations: dict[str, torch.Tensor], n_segments: int, pool: str = "mean"
) -> torch.Tensor:
    # activations: {"Q": (T, L, D_q), "K": (T, L, D_kv), "V": (T, L, D_kv)} ->
    # (T, L, M, 3), channels ordered as PROJECTIONS.
    per_projection = []
    for name in PROJECTIONS:
        raw = activations[name]
        if raw.ndim != 3:
            raise ValueError(
                f"expected {name} raw (T, L, D), got shape {tuple(raw.shape)}"
            )
        # pool in float32: fp16 loses precision over a 128-elem chunk mean
        per_projection.append(pool_segments(raw.float(), n_segments, mode=pool))

    n_tokens = {p.shape[0] for p in per_projection}
    if len(n_tokens) != 1:
        raise ValueError(
            f"Q/K/V disagree on token count: "
            f"{[tuple(p.shape) for p in per_projection]}"
        )

    return torch.stack(per_projection, dim=-1)  # (T, L, M, 3)


def stack_hidden_states(
    hidden_states: tuple[tuple[torch.Tensor, ...], ...],
    n_layers: int | None = None,
    hidden_size: int | None = None,
) -> torch.Tensor:
    # generate_outputs["hidden_states"] (T per-step tuples of (L+1) per-layer
    # (1,1,D) tensors, index 0 = embedding output) -> (T, L, D), dropping the
    # embedding output so L matches the QKV field's layer axis.
    # n_layers/hidden_size are only needed for the empty-hidden_states case
    # (T == 0, e.g. a response fully consumed by run-on truncation).
    if not hidden_states:
        if n_layers is None or hidden_size is None:
            raise ValueError(
                "stack_hidden_states: hidden_states is empty, so n_layers "
                "and hidden_size are required to build a well-formed "
                "(0, L, D) tensor."
            )
        return torch.empty(0, n_layers, hidden_size, dtype=torch.float32)
    per_step = []
    for step in hidden_states:
        per_layer = [layer[0, 0].float() for layer in step[1:]]
        per_step.append(torch.stack(per_layer, dim=0))  # (L, D)
    return torch.stack(per_step, dim=0)  # (T, L, D)


def build_hidden_states_field(
    hidden: torch.Tensor, n_segments: int, pool: str = "mean"
) -> torch.Tensor:
    # Same construction as build_feature_field but one channel instead of
    # three; (T, L, D) -> (T, L, M, 1). Trailing axis kept (not squeezed) so
    # callers that read field.shape[-1] for the channel count work unmodified.
    if hidden.ndim != 3:
        raise ValueError(f"expected hidden (T, L, D), got shape {tuple(hidden.shape)}")
    pooled = pool_segments(hidden.float(), n_segments, mode=pool)  # (T, L, M)
    return pooled.unsqueeze(-1)  # (T, L, M, 1)


def pool_layer_axis(field: torch.Tensor, n_layers_out: int) -> torch.Tensor:
    # Adaptive mean-pool the LAYER axis to a fixed size, for cross-LLM work
    # where models differ in L. No-op when L == n_layers_out. LOSSY and
    # non-invertible -- steering runs must leave l_eff unset.
    n_tok, n_layers, n_segments, n_chan = field.shape
    if n_layers == n_layers_out:
        return field
    if n_layers < n_layers_out:
        raise ValueError(
            f"cannot pool layer axis up: have L={n_layers}, asked for {n_layers_out}"
        )
    flat = field.permute(0, 2, 3, 1).reshape(n_tok * n_segments * n_chan, 1, n_layers)
    pooled = torch.nn.functional.adaptive_avg_pool1d(flat, n_layers_out)
    return pooled.reshape(n_tok, n_segments, n_chan, n_layers_out).permute(0, 3, 1, 2)
