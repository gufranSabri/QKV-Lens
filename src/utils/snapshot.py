"""Copies the code that produced a run into the run's own directory.

Configs and CLI flags are recorded elsewhere (config.json), but the code
itself changes over time. Without a snapshot, reproducing an old run means
trusting that the right git commit is still checked out.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.config import REPO_ROOT

#: Copied verbatim into every run's code/ snapshot.
SNAPSHOT_PATHS = ("src", "main.py", "configs")

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def snapshot_code(run_dir: Path) -> None:
    dest = run_dir / "code"
    dest.mkdir(parents=True, exist_ok=True)

    for name in SNAPSHOT_PATHS:
        src = REPO_ROOT / name
        if src.is_dir():
            shutil.copytree(src, dest / name, ignore=_IGNORE, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, dest / name)
