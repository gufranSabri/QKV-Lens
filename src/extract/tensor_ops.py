"""Pure tensor math for building per-token QKV feature fields.

This module is deliberately free of any model/HuggingFace dependency so the
feature-map construction can be unit-tested without loading an LLM.

QKV-Lens Algorithm 1, verbatim. For one example:

    Q/K/V raw   (T, L, D_c)  per-token, per-layer projection vectors
      -> mean_pool_segments  ->  (T, L, M)   for each of c in {Q, K, V}
      -> stack on a channel axis  ->  (T, L, M, 3)

The trailing axis is the PROJECTION axis: channel 0 = Q, 1 = K, 2 = V. That is
the paper's `F_{t,l,m,c}` exactly (Eq. 4-7), and it is the coordinate system the
steering stage writes back into -- see src/steer/ for the inverse (broadcast)
direction.

Mean pooling over M = 32 contiguous segments is the paper's setting (§5.3, and
Table 3, where mean beats max and strided at matched resolution). Nothing else
is offered: an alternative pooling rule would change what `F` means and would
therefore change what a steering coordinate refers to.
"""

from __future__ import annotations

import torch

#: Channel order of the trailing axis of every feature field this module builds.
#: This ordering is load-bearing -- it is what makes `F[..., c]` addressable by
#: projection name, both when the detector reads a field and when the steerer
#: writes one back.
PROJECTIONS = ("Q", "K", "V")


def mean_pool_segments(raw: torch.Tensor, n_segments: int) -> torch.Tensor:
    """Reduce the feature axis D to `n_segments` by mean-pooling contiguous chunks.

    Args:
        raw:        (..., D) tensor, typically (T, L, D).
        n_segments: number of output segments M. D must be divisible by M.

    Returns:
        (..., n_segments)

    Segment m covers `raw[..., m*S : (m+1)*S]` where S = D // M, matching the
    paper's Eq. 4:  q_{t,l,m} = (1/S) * sum_j Q^{(l)}_{t,(m-1)S+j}.

    Divisibility is required, not worked around: a ragged final segment would
    average a different number of dimensions than the others, so segment m would
    no longer mean the same thing across m -- and the steering broadcast assumes
    every segment covers exactly S dimensions.

    For Llama-3-8B the Q projection has D=4096 (32 heads x head_dim 128), so at
    M=32 the segments coincide exactly with attention heads and segment m is
    "the mean activation of head m". That alignment is a happy accident of the
    architecture, NOT something enforced -- under GQA the K/V projections have
    D=1024 and pooling those to 32 segments splits each kv-head across 4.
    """
    d = raw.shape[-1]
    if n_segments <= 0:
        raise ValueError(f"n_segments must be positive, got {n_segments}")
    if d % n_segments != 0:
        raise ValueError(
            f"feature dim D={d} is not divisible by n_segments={n_segments}; "
            "pooling would drop or duplicate dimensions"
        )

    segment_size = d // n_segments
    chunked = raw.reshape(*raw.shape[:-1], n_segments, segment_size)
    return chunked.mean(dim=-1)


def build_feature_field(
    activations: dict[str, torch.Tensor], n_segments: int
) -> torch.Tensor:
    """Q/K/V raw activations -> the paper's feature field (T, L, M, 3).

    Args:
        activations: {"Q": (T, L, D_q), "K": (T, L, D_kv), "V": (T, L, D_kv)}.
            K and V are narrower than Q under GQA; each is pooled to the same
            M independently, so the three stack cleanly.
        n_segments:  M, the number of pooled segments per layer.

    Returns:
        (T, L, M, 3), channels ordered as `PROJECTIONS` (Q, K, V).
    """
    per_projection = []
    for name in PROJECTIONS:
        raw = activations[name]
        if raw.ndim != 3:
            raise ValueError(
                f"expected {name} raw (T, L, D), got shape {tuple(raw.shape)}"
            )
        # Pool in float32: the captured activations may be fp16, where a mean
        # over a 128-element chunk loses precision at the top of the range.
        # Cast back at save time.
        per_projection.append(mean_pool_segments(raw.float(), n_segments))

    n_tokens = {p.shape[0] for p in per_projection}
    if len(n_tokens) != 1:
        raise ValueError(
            f"Q/K/V disagree on token count: "
            f"{[tuple(p.shape) for p in per_projection]}"
        )

    return torch.stack(per_projection, dim=-1)  # (T, L, M, 3)


def pool_layer_axis(field: torch.Tensor, n_layers_out: int) -> torch.Tensor:
    """Down-pool the LAYER axis of a feature field to a fixed size.

    Args:
        field:        (T, L, M, 3)
        n_layers_out: target number of layers L_eff.

    Returns:
        (T, L_eff, M, 3)

    Needed only for cross-LLM work: Llama-3-8B has L=32 while Qwen2.5-7B has
    L=28, so their fields are 32xM and 28xM and a single CNN cannot consume
    both. Pooling the layer axis to a common L_eff makes the field's height a
    fixed hyperparameter rather than a property of the LLM.

    Uses adaptive mean pooling, so L need not be divisible by L_eff. This is a
    no-op when L == n_layers_out, so it is safe to call unconditionally.

    NOTE for steering: this is a LOSSY, non-invertible remap of the layer axis.
    A field pooled to L_eff != L cannot be written back to specific LLM layers,
    so steering runs must leave `l_eff` unset.
    """
    n_tok, n_layers, n_segments, n_chan = field.shape
    if n_layers == n_layers_out:
        return field
    if n_layers < n_layers_out:
        raise ValueError(
            f"cannot pool layer axis up: have L={n_layers}, asked for {n_layers_out}"
        )
    # adaptive_avg_pool1d wants (N, C, L_in) and pools the last axis.
    flat = field.permute(0, 2, 3, 1).reshape(n_tok * n_segments * n_chan, 1, n_layers)
    pooled = torch.nn.functional.adaptive_avg_pool1d(flat, n_layers_out)
    return pooled.reshape(n_tok, n_segments, n_chan, n_layers_out).permute(0, 3, 1, 2)
