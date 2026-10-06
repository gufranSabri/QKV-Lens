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

#: Segment-pooling strategies, for the pooling ablation (QKV-Lens Table 3).
#: `mean` is the paper's canonical, fixed setting (see module docstring) --
#: the others exist ONLY to reproduce that ablation, never for steering: a
#: field built under max/strided is not the coordinate system the steering
#: stage's broadcast assumes.
#:   mean     average of each contiguous S-wide chunk (the paper's Eq. 4).
#:   max      peak value of each chunk.
#:   strided  every Sth dimension, i.e. raw[..., 0], raw[..., S], raw[..., 2S],
#:            ... -- true subsampling, not a pooled reduction.
POOL_MODES = ("mean", "max", "strided")


def mean_pool_segments(raw: torch.Tensor, n_segments: int) -> torch.Tensor:
    """Reduce the feature axis D to `n_segments` by mean-pooling contiguous chunks.

    Thin wrapper around `pool_segments(raw, n_segments, mode="mean")`, kept as
    its own name because `mean` is the paper's one fixed, non-ablated setting
    (see module docstring) -- every call site that builds a real feature field
    for training/steering should read as "the canonical pooling", not as one
    mode among several.
    """
    return pool_segments(raw, n_segments, mode="mean")


def pool_segments(raw: torch.Tensor, n_segments: int, mode: str = "mean") -> torch.Tensor:
    """Reduce the feature axis D to `n_segments` by pooling or subsampling.

    Args:
        raw:        (..., D) tensor, typically (T, L, D).
        n_segments: number of output segments M. D must be divisible by M.
        mode:       one of POOL_MODES. `mean` is the paper's fixed setting
                    (Eq. 4); `max` and `strided` exist only for the pooling
                    ablation (QKV-Lens Table 3) -- see POOL_MODES.

    Returns:
        (..., n_segments)

    For the contiguous modes (mean, max), segment m covers
    `raw[..., m*S : (m+1)*S]` where S = D // M. `strided` instead takes
    `raw[..., m*S]` -- one representative dimension per stride, not an
    aggregate -- so its segment m means something different from the other
    two modes' segment m even though the shapes match.

    Divisibility is required, not worked around: a ragged final segment would
    cover a different number of source dimensions than the others, so segment
    m would no longer mean the same thing across m -- and the steering
    broadcast (mean/max modes only -- see `mean_pool_segments`) assumes every
    segment covers exactly S dimensions.

    For Llama-3-8B the Q projection has D=4096 (32 heads x head_dim 128), so at
    M=32 the segments coincide exactly with attention heads and segment m is
    "the mean/max/first-element activation of head m". That alignment is a
    happy accident of the architecture, NOT something enforced -- under GQA
    the K/V projections have D=1024 and pooling those to 32 segments splits
    each kv-head across 4.
    """
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
        # Every Sth dimension, starting at 0: a true subsample, not a
        # reduction over each chunk.
        return raw[..., ::segment_size][..., :n_segments]

    chunked = raw.reshape(*raw.shape[:-1], n_segments, segment_size)
    if mode == "max":
        return chunked.amax(dim=-1)
    return chunked.mean(dim=-1)


def build_feature_field(
    activations: dict[str, torch.Tensor], n_segments: int, pool: str = "mean"
) -> torch.Tensor:
    """Q/K/V raw activations -> the paper's feature field (T, L, M, 3).

    Args:
        activations: {"Q": (T, L, D_q), "K": (T, L, D_kv), "V": (T, L, D_kv)}.
            K and V are narrower than Q under GQA; each is pooled to the same
            M independently, so the three stack cleanly.
        n_segments:  M, the number of pooled segments per layer.
        pool:        one of POOL_MODES. `mean` (default) is the paper's fixed
            setting; `max`/`strided` exist only for the pooling ablation
            (QKV-Lens Table 3) and produce a field OUTSIDE the coordinate
            system the steering stage assumes -- see POOL_MODES.

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
    """generate_outputs["hidden_states"] (capture_all's shape, matching
    transformers' GenerateDecoderOnlyOutput.hidden_states exactly) -> (T, L, D).

    Args:
        hidden_states: tuple of T per-step tuples, each of (L+1) per-layer
            tensors shaped (1, 1, D) -- index 0 of each step is the embedding
            output, BEFORE any transformer layer; see
            hallushift_features.plot_internal_state_2's docstring for the
            same convention. Dropped here (sliced to [1:]) so the returned
            L axis means "output of transformer layer l", the same thing the
            QKV field's L axis means -- keeping the two fields' L coordinate
            comparable is the whole point of extracting hidden states through
            the SAME shared generation pass as QKV.
        n_layers, hidden_size: the model's own geometry (ModelGeometry.n_layers
            / .hidden_size), used ONLY for the T=0 case below -- an empty
            `hidden_states` tuple still needs a well-formed (0, L, D) return
            (not just a bare torch.empty(0)), because downstream
            build_hidden_states_field, and the QKV path's analogous
            build_feature_field, both expect every raw tensor to be
            3-dimensional even when T happens to be 0 (a response fully
            consumed by run-on truncation -- see run_extraction.py's
            `n_keep == 0` case). Required whenever `hidden_states` is empty;
            omit them only when it is known to be non-empty.

    Returns:
        (T, L, D) float32, L = the model's transformer layer count (not L+1).
    """
    if not hidden_states:
        if n_layers is None or hidden_size is None:
            raise ValueError(
                "stack_hidden_states: hidden_states is empty, so n_layers "
                "and hidden_size are required to build a well-formed "
                "(0, L, D) tensor (see this function's docstring)."
            )
        return torch.empty(0, n_layers, hidden_size, dtype=torch.float32)
    per_step = []
    for step in hidden_states:
        # step[0] is the embedding layer; step[1:] are the L transformer
        # layers' outputs, each (1, 1, D) -- squeeze to (D,).
        per_layer = [layer[0, 0].float() for layer in step[1:]]
        per_step.append(torch.stack(per_layer, dim=0))  # (L, D)
    return torch.stack(per_step, dim=0)  # (T, L, D)


def build_hidden_states_field(
    hidden: torch.Tensor, n_segments: int, pool: str = "mean"
) -> torch.Tensor:
    """Hidden-state activations -> a (T, L, M, 1) field, the same construction
    as build_feature_field but with ONE channel instead of three.

    Args:
        hidden:      (T, L, D) -- see stack_hidden_states.
        n_segments:  M, the number of pooled segments per layer.
        pool:        one of POOL_MODES; see build_feature_field.

    Returns:
        (T, L, M, 1). The trailing axis is kept (not squeezed) so a
        hidden-states field has the SAME rank as a QKV field -- every caller
        that indexes `field[..., c]` or reads field.shape[-1] for the channel
        count (e.g. QKVFieldDataset) keeps working unmodified; only the
        channel count itself (1 vs 3) differs, which those callers already
        read dynamically from geometry.json rather than assuming 3 -- see
        src/data/dataset.py's `n_channels` construction-time argument.
    """
    if hidden.ndim != 3:
        raise ValueError(f"expected hidden (T, L, D), got shape {tuple(hidden.shape)}")
    pooled = pool_segments(hidden.float(), n_segments, mode=pool)  # (T, L, M)
    return pooled.unsqueeze(-1)  # (T, L, M, 1)


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
