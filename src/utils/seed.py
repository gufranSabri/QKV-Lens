"""Reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic cuDNN costs a little speed but makes runs comparable, which
    # matters more than throughput when the whole point is an ablation table.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(cuda_idx: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_idx}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
