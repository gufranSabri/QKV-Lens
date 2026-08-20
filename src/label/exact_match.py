"""Correctness scoring by string matching.

Adapted from ACT-ViT's `utils/generation_utils.py`, which in turn follows
LLMsKnow (technion-cs-nlp/LLMsKnow). We keep their protocol so our numbers are
comparable to theirs.

Every function returns `correct` in {0, 1}. The hallucination label is the
complement:  label = 1 - correct  (1 = hallucinated).

WHY WE RE-ANNOTATE AT ALL
-------------------------
The dataset ships a gold answer, not a hallucination label. Once the LLM writes
its OWN response, whether that response is a hallucination is a property of the
generated text -- so the label has to be recomputed against the gold answer.
Both baselines do exactly this; the schemes differ only in how they compare.
"""

from __future__ import annotations

import ast


def _as_list(value) -> list[str]:
    """Normalise a gold field into a list of acceptable answer strings."""
    if value is None:
        return []
    if isinstance(value, str):
        # TriviaQA aliases sometimes arrive as a stringified list.
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed]
            except (ValueError, SyntaxError):
                pass
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def correctness_substring(answer: str, gold) -> int:
    """Correct if ANY gold answer appears anywhere in the response.

    Used for triviaqa (many aliases) and truthfulqa. Case-insensitive.
    This is a lenient criterion -- it rewards a response that contains the right
    answer even when buried in surrounding text, which is what these baselines do
    (the models are prompted to answer concisely, so the response is short).
    """
    if not answer:
        return 0
    haystack = answer.lower()
    for candidate in _as_list(gold):
        needle = str(candidate).lower().strip()
        if needle and needle in haystack:
            return 1
    return 0


CORRECTNESS_FN = {
    "triviaqa": correctness_substring,
    "truthfulqa": correctness_substring,
}


def score_exact_match(dataset_name: str, answer: str, gold) -> tuple[float, int]:
    """Returns (score, label). score is the 0/1 correctness; label = 1 - score."""
    if dataset_name not in CORRECTNESS_FN:
        raise KeyError(
            f"no exact-match scorer for dataset {dataset_name!r}. "
            f"Known: {sorted(CORRECTNESS_FN)}"
        )
    correct = CORRECTNESS_FN[dataset_name](answer, gold)
    return float(correct), 1 - correct
