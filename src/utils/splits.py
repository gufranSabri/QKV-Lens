# Stratified train/val/test split, persisted so runs are comparable. The test
# slice is carved first, then val is a stratified slice of what remains, so
# early stopping never sees the test set.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_split(
    labels: list[int],
    val_fraction: float = 0.1,
    test_fraction: float = 0.25,
    seed: int = 0,
    cache: Path | None = None,
) -> tuple[list[int], list[int], list[int]]:
    # val_fraction is a fraction of the non-test examples, test_fraction of all of them.
    key = {"n": len(labels), "seed": seed,
           "val_fraction": val_fraction, "test_fraction": test_fraction}

    if cache is not None and cache.exists():
        try:
            data = json.loads(cache.read_text())
        except json.JSONDecodeError:  # truncated by a job killed mid-write
            data = {}
        # Compare the full parameterisation, not just (n, seed): a run that
        # changes only test_fraction must not silently reuse a stale split.
        if all(data.get(k) == v for k, v in key.items()):
            logger.info("reusing cached split from %s", cache)
            return data["train"], data["val"], data["test"]
        logger.warning("cached split at %s is stale; rebuilding", cache)

    idx = np.arange(len(labels))
    y = np.asarray(labels)

    def _stratify(subset_y):
        if len(np.unique(subset_y)) > 1:
            return subset_y
        logger.warning("only one class present; falling back to an unstratified split")
        return None

    if test_fraction > 0:
        rest, test_arr = train_test_split(
            idx, test_size=test_fraction, random_state=seed, stratify=_stratify(y)
        )
    else:
        rest, test_arr = idx, idx[:0]
    train_arr, val_arr = train_test_split(
        rest, test_size=val_fraction, random_state=seed, stratify=_stratify(y[rest])
    )
    train, val, test = (sorted(a.tolist()) for a in (train_arr, val_arr, test_arr))

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps({**key, "train": train, "val": val, "test": test})
        )
        logger.info("saved split to %s", cache)

    return train, val, test
