# Dispatch to a labeling scheme, so relabeling never requires re-extraction.

from __future__ import annotations

from pathlib import Path

from src.config import Config
from src.label.bleurt import score_bleurt_batch
from src.label.exact_match import score_exact_match
from src.label.llm_judge import score_llm_judge_batch


def label_examples(
    cfg: Config,
    answers: list[str],
    golds: list,
    cache_dir: Path | None = None,
) -> list[tuple[float, int]]:
    # Returns [(score, label), ...]; label == 1 means HALLUCINATED.
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

    if scheme in ("llm_judge", "ollama_judge"):
        local = scheme == "ollama_judge"
        return score_llm_judge_batch(
            answers,
            golds,
            model_name=cfg.labeling.ollama_model if local else cfg.labeling.llm_judge_model,
            cache_path=cache_dir / "judge_cache.jsonl" if cache_dir is not None else None,
            backend="ollama" if local else "novita",
        )

    raise ValueError(f"unknown labeling scheme {scheme!r}")
