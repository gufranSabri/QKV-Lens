# Correctness scoring by string matching. Adapted from ACT-ViT's protocol
# (via LLMsKnow) for comparable numbers. label = 1 - correct (1 = hallucinated).

from __future__ import annotations

import ast


def _as_list(value) -> list[str]:
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
    # Correct if ANY gold answer appears anywhere in the response (case-insensitive).
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
    if dataset_name not in CORRECTNESS_FN:
        raise KeyError(
            f"no exact-match scorer for dataset {dataset_name!r}. "
            f"Known: {sorted(CORRECTNESS_FN)}"
        )
    correct = CORRECTNESS_FN[dataset_name](answer, gold)
    return float(correct), 1 - correct
