"""Dispatch to a labeling scheme, so relabeling never requires re-extraction."""

from __future__ import annotations

from src.config import Config
from src.label.bleurt import score_bleurt_batch
from src.label.exact_match import score_exact_match


def label_examples(
    cfg: Config,
    answers: list[str],
    golds: list,
) -> list[tuple[float, int]]:
    """Score a batch of generated answers. Returns [(score, label), ...].

    label == 1 means HALLUCINATED.
    """
    scheme = cfg.labeling.scheme

    if scheme == "exact_match":
        return [
            score_exact_match(cfg.dataset.name, ans, gold)
            for ans, gold in zip(answers, golds)
        ]

    if scheme == "bleurt":
        return score_bleurt_batch(
            answers,
            golds,
            checkpoint=cfg.labeling.bleurt_checkpoint,
            threshold=cfg.labeling.bleurt_threshold,
        )

    raise ValueError(f"unknown labeling scheme {scheme!r}")
