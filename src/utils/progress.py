"""Progress bar wrapper around tqdm."""

from __future__ import annotations

from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def progress(iterable: Iterable[T], **tqdm_kwargs) -> Iterator[T]:
    from tqdm import tqdm

    return tqdm(iterable, **tqdm_kwargs)
