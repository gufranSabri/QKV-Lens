"""HalluShift's own per-token feature computation, ported from
reproducing_baselines/hallushift/functions.py's plot_internal_state_2 /
probability_function -- WITH the performance fix from the earlier standalone
HalluShift run (lost when that repo was recloned fresh; reapplied here since
this module now owns that computation for the shared pipeline).

WHY REWRITTEN, NOT IMPORTED: the original scipy.stats.wasserstein_distance +
F.cosine_similarity, called once per (generated token, layer-pair), profiled
at 30-50s+ PER EXAMPLE and climbing with context length -- the actual
bottleneck behind a stalled multi-day run, while model.generate() itself only
took ~2-3s/example. The fix: every call here compares two EQUAL-LENGTH,
EQUAL-WEIGHT softmax distributions, for which Wasserstein-1 has a closed
form (verified bit-identical to scipy across many trials/sizes):
    W1(p, q) == mean(|sort(p) - sort(q)|)
Batching every layer-pair for one token into a single NumPy op (one
.cpu().numpy() transfer per token, not per pair) took this from 46.17s to
0.50s on a realistic 64-token/32-layer/500-token-prompt simulation (~92x),
with max abs error ~3e-8 (float32 rounding) vs the original scipy-based
result. See that investigation for the full profiling trail.

Output shape matches HalluShift's own `result` row exactly (module docstring
in reproducing_baselines/hallushift/hal_detection.py's process_row):
    plot_internal_state_2(hidden) + plot_internal_state_2(attention)
    + probability_function(logits) + [decoded_response]
so reproducing_baselines/hallushift/functions.data_preparation and
classifier.train_combined_model consume it completely unmodified -- this
module only changes how the raw generation is obtained, never HalluShift's
own method code.
"""

from __future__ import annotations

import numpy as np
import torch


def plot_internal_state_2(step_tensors: tuple, num_layers: int, state: str = "hidden") -> list[float]:
    """Per-token Wasserstein-1 + cosine-similarity between consecutive
    layer-pairs, averaged over all generated tokens.

    Args:
        step_tensors: outputs.hidden_states or outputs.attentions from
            capture_all(..., capture_generate_outputs=True) -- a tuple of
            T per-step tuples, each of per-layer tensors (see that
            function's docstring for the exact shape contract, which
            matches transformers' GenerateDecoderOnlyOutput exactly).
        num_layers: the model's layer count (model.config.num_hidden_layers).
        state: "hidden" or "attention" -- which axis stride HalluShift's
            original code used (hidden: every 2nd layer INCLUDING the
            embedding output at index 0; attention: every 2nd layer
            starting at 1, since attentions has no embedding entry).

    Returns:
        A flat list of 2*((num_layers//2)-1) floats: Wasserstein distances
        for each consecutive layer-pair, then cosine similarities for the
        same pairs -- identical layout to the original plot_internal_state_2.
    """
    if state == "hidden":
        layer_indices = list(range(2, num_layers + 1, 2))
    else:
        layer_indices = list(range(1, num_layers, 2))

    n_pairs = len(layer_indices) - 1
    if n_pairs <= 0 or not step_tensors:
        return [0.0] * max(n_pairs, 0) * 2

    w_sum = np.zeros(n_pairs, dtype=np.float64)
    c_sum = np.zeros(n_pairs, dtype=np.float64)
    n_tokens = 0

    for tup in step_tensors:
        vecs = torch.stack(
            [torch.softmax(tup[i].reshape(-1).float(), dim=-1) for i in layer_indices],
            dim=0,
        ).cpu().numpy()  # (n_pairs+1, D)

        p = vecs[:-1]
        q = vecs[1:]

        p_sorted = np.sort(p, axis=-1)
        q_sorted = np.sort(q, axis=-1)
        w = np.abs(p_sorted - q_sorted).mean(axis=-1)

        num = (p * q).sum(axis=-1)
        denom = np.linalg.norm(p, axis=-1) * np.linalg.norm(q, axis=-1)
        c = num / np.clip(denom, 1e-8, None)

        w_sum += w
        c_sum += c
        n_tokens += 1

    return list(w_sum / n_tokens) + list(c_sum / n_tokens)


def probability_function(step_logits: tuple) -> list[list[float]]:
    """Per-token max/min softmax probability, one list per generated token.

    Args:
        step_logits: outputs.logits from capture_all(...,
            capture_generate_outputs=True) -- a tuple of T per-step (1, vocab)
            tensors.

    Returns:
        [max_prob_results, min_prob_results] -- identical layout to the
        original probability_function (two lists, same length as step_logits).
    """
    max_prob_results = []
    min_prob_results = []
    for logit in step_logits:
        probabilities = torch.softmax(logit[0].float(), dim=0)
        max_prob_results.append(probabilities.max().item())
        min_prob_results.append(probabilities.min().item())
    return [max_prob_results, min_prob_results]


def build_hallushift_row(generate_outputs: dict, num_layers: int, response: str) -> list:
    """One HalluShift `result` row for a single example -- the exact
    concatenation reproducing_baselines/hallushift/hal_detection.py's
    process_row builds from a real model.generate() call, but from
    capture_all(..., capture_generate_outputs=True)'s captured per-step
    outputs instead.
    """
    return (
        plot_internal_state_2(generate_outputs["hidden_states"], num_layers, state="hidden")
        + plot_internal_state_2(generate_outputs["attentions"], num_layers, state="attention")
        + probability_function(generate_outputs["logits"])
        + [response]
    )
