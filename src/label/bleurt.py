"""Correctness scoring by BLEURT semantic similarity (HalluShift's protocol).

HalluShift scores the generated answer against every reference answer with
BLEURT-20-D12 and takes the MAXIMUM, then thresholds:

    hallucination = 1 if max_score <= threshold else 0     (threshold = 0.5)

This is softer than exact matching -- it credits a correct paraphrase that
substring matching would mark wrong -- at the cost of a heavyweight dependency.

BLEURT is a TensorFlow model and is NOT installed by default here. Install it
only if you want `labeling.scheme: bleurt`:

    pip install --upgrade pip
    git clone https://github.com/google-research/bleurt.git && pip install ./bleurt
    wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip
    unzip BLEURT-20-D12.zip -d models/

The scorer is loaded lazily and cached, because loading it costs several seconds
and we score thousands of examples per run.
"""

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
            "installed. Either install it (see src/label/bleurt.py docstring) or "
            "use labeling.scheme='exact_match'."
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
    """Returns (max_bleurt_score, label). label = 1 (hallucinated) if score <= threshold.

    An empty answer or an empty-string reference is still scored by BLEURT, not
    hard-labelled -- HalluShift applies no such guard (bleurt_processing just
    thresholds whatever score comes back, functions.py:166). Only a truly empty
    reference LIST (no gold answers at all) has nothing to score against.
    """
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
    """Batched variant -- one BLEURT call for the whole corpus.

    BLEURT's per-call overhead is large, so scoring 10k examples one at a time is
    far slower than flattening every (candidate, reference) pair into a single
    call and then taking the max per example. Same result, much faster.
    """
    scorer = _get_scorer(checkpoint)

    flat_cands: list[str] = []
    flat_refs: list[str] = []
    spans: list[tuple[int, int]] = []  # (start, end) into the flat lists, per example

    for answer, gold in zip(answers, golds):
        refs = _as_list(gold)
        start = len(flat_cands)
        # Score even an empty answer/reference -- only a genuinely empty
        # reference LIST has nothing to compare against. See score_bleurt.
        if refs:
            flat_cands.extend([answer] * len(refs))
            flat_refs.extend(refs)
        spans.append((start, len(flat_cands)))

    # Score in chunks with a progress bar rather than one giant call. A single
    # scorer.score() over the whole corpus (tens of thousands of pairs for a 10k
    # dataset with several gold answers each) prints NOTHING for many minutes and
    # is indistinguishable from a hang -- and holding that many activations at
    # once can spike GPU memory. Chunking bounds the memory and shows progress.
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
        if start == end:  # no refs at all
            out.append((0.0, 1))
        else:
            best = float(max(flat_scores[start:end]))
            out.append((best, int(best <= threshold)))
    return out
