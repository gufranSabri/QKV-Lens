# Correctness scoring by BLEURT semantic similarity (HalluShift's protocol):
# score the answer against every reference with BLEURT-20-D12, take the max,
# threshold at 0.5. Not installed by default -- see scripts/install.sh --bleurt.

from __future__ import annotations

from functools import lru_cache

from src.label.exact_match import _as_list


@lru_cache(maxsize=2)
def _get_scorer(checkpoint: str):
    try:
        from bleurt import score as bleurt_score
    except ImportError as exc:
        raise ImportError(
            "labeling.scheme='bleurt' requires the BLEURT package, which is not "
            "installed. Either run scripts/install.sh --bleurt or use "
            "labeling.scheme='exact_match'."
        ) from exc

    import os

    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(
            f"BLEURT checkpoint not found at {checkpoint!r}. Download BLEURT-20-D12 "
            "and unzip it there, or set labeling.bleurt_checkpoint."
        )
    return bleurt_score.BleurtScorer(checkpoint)


def score_bleurt(
    answer: str,
    gold,
    checkpoint: str = "models/BLEURT-20-D12",
    threshold: float = 0.5,
) -> tuple[float, int]:
    # Returns (max_bleurt_score, label); label = 1 if score <= threshold.
    refs = _as_list(gold)
    if not refs:
        return 0.0, 1

    scorer = _get_scorer(checkpoint)
    scores = scorer.score(references=refs, candidates=[answer] * len(refs))
    best = float(max(scores))
    return best, int(best <= threshold)


def score_bleurt_batch(
    answers: list[str],
    golds: list,
    checkpoint: str = "models/BLEURT-20-D12",
    threshold: float = 0.5,
) -> list[tuple[float, int]]:
    # Flattens every (candidate, reference) pair into one scorer call instead
    # of scoring examples one at a time -- much faster for a full corpus.
    scorer = _get_scorer(checkpoint)

    flat_cands: list[str] = []
    flat_refs: list[str] = []
    spans: list[tuple[int, int]] = []

    for answer, gold in zip(answers, golds):
        refs = _as_list(gold)
        start = len(flat_cands)
        if refs:
            flat_cands.extend([answer] * len(refs))
            flat_refs.extend(refs)
        spans.append((start, len(flat_cands)))

    # Chunked (not one giant call) so progress is visible and memory is bounded.
    flat_scores: list[float] = []
    if flat_cands:
        from src.utils.logger import get_logger
        from src.utils.progress import progress

        _log = get_logger(__name__)
        n_pairs = len(flat_cands)
        _log.info("BLEURT: scoring %d (candidate, reference) pairs", n_pairs)
        chunk = 512
        for i in progress(range(0, n_pairs, chunk), desc="BLEURT scoring", ncols=100):
            flat_scores.extend(
                scorer.score(
                    references=flat_refs[i : i + chunk],
                    candidates=flat_cands[i : i + chunk],
                )
            )

    out = []
    for start, end in spans:
        if start == end:
            out.append((0.0, 1))
        else:
            best = float(max(flat_scores[start:end]))
            out.append((best, int(best <= threshold)))
    return out
