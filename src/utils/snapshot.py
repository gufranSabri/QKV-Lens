# Copies the code that produced a run into the run's own directory, so
# reproducing an old run doesn't depend on the right git commit still being checked out.

from __future__ import annotations

import shutil
from pathlib import Path

from src.config import REPO_ROOT

SNAPSHOT_PATHS = ("src", "detector.py", "configs")

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
