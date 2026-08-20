"""Stratified train/test split, persisted so runs are comparable.

Mirrors HalluShift's classifier.train_combined_model: a single stratified
sklearn.train_test_split(test_size=..., stratify=y, random_state=seed) into
train and test, with the *same* test set doubling as the early-stopping
validation signal (see hallushift/classifier.py:104-138). There is no
separate held-out set -- `val` and `test` below are the same indices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_split(
    labels: list[int],
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 0,
    cache: Path | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Stratified split of indices into (train, val, test), val == test.

    Stratified because hallucination rates are often far from 50/50; a random
    split could hand the eval set a wildly different positive rate and make
    its AUROC incomparable to train.

    Every dataset here has a single upstream split (see datasets.SPLIT_SOURCES),
    so the only honest test set is one we carve out ourselves -- 25%, matching
    the paper's 75/25 ratio and HalluShift's `test_size`. `test_fraction=0`
    would yield an empty eval split and fall back to `val_fraction`; it is kept
    only so the function stays total.

    Persisted to `cache` so that repeated runs (and the model-selection decisions
    they drive) all see the same split.
    """
    key = {"n": len(labels), "seed": seed,
           "val_fraction": val_fraction, "test_fraction": test_fraction}

    if cache is not None and cache.exists():
        data = json.loads(cache.read_text())
        # Compare the full parameterisation, not just (n, seed): a run that
        # changes only test_fraction must not silently reuse a stale split.
        if all(data.get(k) == v for k, v in key.items()):
            logger.info("reusing cached split from %s", cache)
            return data["train"], data["val"], data.get("test", [])
        logger.warning("cached split at %s is stale; rebuilding", cache)

    idx = np.arange(len(labels))
    y = np.asarray(labels)

    def _stratify(subset_y):
        if len(np.unique(subset_y)) > 1:
            return subset_y
        logger.warning("only one class present; falling back to an unstratified split")
        return None

    # test_fraction is the eval proportion; val_fraction is the fallback if it
    # is zero. Same role as HalluShift's test_size either way.
    eval_fraction = test_fraction if test_fraction > 0 else val_fraction
    if eval_fraction <= 0:
        train = sorted(idx.tolist())
        eval_idx: list[int] = []
    else:
        train_arr, eval_arr = train_test_split(
            idx, test_size=eval_fraction, random_state=seed, stratify=_stratify(y)
        )
        train, eval_idx = sorted(train_arr.tolist()), sorted(eval_arr.tolist())

    # HalluShift reuses the same held-out slice for early stopping and for the
    # final reported metric -- no independent validation set.
    val, test = eval_idx, eval_idx

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps({**key, "train": train, "val": val, "test": test})
        )
        logger.info("saved split to %s", cache)

    return train, val, test
