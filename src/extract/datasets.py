"""Text dataset loading: prompt construction and gold-answer extraction.

Each loader returns a list of Example(prompt, gold), where `gold` is whatever the
labeling scheme needs to judge correctness (a string, or a list of acceptable
aliases). Prompt templates follow HalluShift/ACT-ViT so responses are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Example:
    prompt: str
    gold: object   # str, or list[str] of acceptable answers
    idx: int


def load_triviaqa(cfg, n: int, split: str = "validation") -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("trivia_qa", "rc.nocontext", split=split)

    # Deduplicate by question_id -- TriviaQA repeats questions across contexts,
    # and HalluShift drops the duplicates before generating.
    seen: set = set()
    out: list[Example] = []
    for row in ds:
        if len(out) >= n:
            break
        qid = row["question_id"]
        if qid in seen:
            continue
        seen.add(qid)
        out.append(
            Example(
                prompt=cfg.dataset.prompt_template.format(question=row["question"]),
                # All aliases count as correct -- "JFK" and "John F. Kennedy".
                gold=list(row["answer"]["aliases"]) or [row["answer"]["value"]],
                idx=len(out),
            )
        )
    return out


def load_truthfulqa(cfg, n: int) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "generation", split="validation")
    out = []
    for row in ds:
        if len(out) >= n:
            break
        # `best_answer` ONLY -- HalluShift discards `correct_answers` when
        # building references (hal_detection.py:316). Including them would make
        # our labels strictly more lenient than theirs, and the AUROCs would no
        # longer be measuring the same task.
        #
        # Unconditionally `[best_answer]`, even if it's falsy -- HalluShift's
        # `.apply(lambda row: [row])` (hal_detection.py:317) never drops it
        # either; an empty best_answer still becomes a one-element reference
        # list and gets BLEURT-scored like anything else, not hard-labelled.
        out.append(
            Example(
                prompt=cfg.dataset.prompt_template.format(question=row["question"]),
                gold=[row["best_answer"]],
                idx=len(out),
            )
        )
    return out


def _coqa_rows(split: str) -> list[dict]:
    """Download CoQA and flatten each dialogue turn into its own row.

    Mirrors HalluShift (hal_detection.py:81-128): the story ACCUMULATES the
    preceding Q/A pairs, so turn k is conditioned on the dialogue so far. Getting
    this wrong would make our contexts -- and therefore the task -- different.
    """
    import json
    import urllib.request
    from pathlib import Path

    from src.config import REPO_ROOT

    fname = "coqa-train-v1.0.json" if split == "train" else "coqa-dev-v1.0.json"
    dest = Path(REPO_ROOT) / "data" / "raw" / "coqa" / fname
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://downloads.cs.stanford.edu/nlp/data/coqa/{fname}"
        urllib.request.urlretrieve(url, dest)

    rows: list[dict] = []
    for sample in json.loads(dest.read_text())["data"]:
        story = sample["story"]
        for i, question in enumerate(sample["questions"]):
            answer = sample["answers"][i]["input_text"]
            rows.append({"story": story, "question": question["input_text"], "answer": answer})
            # Append this turn to the running context for the NEXT question.
            story += f' Q: {question["input_text"]} A: {answer}'
            if story and story[-1] != ".":
                story += "."
    return rows


def load_coqa(cfg, n: int, split: str = "dev") -> list[Example]:
    """Context (the accumulating story) is NOT truncated -- HalluShift's
    `truncate_after_words` is defined but never called for any dataset."""
    out = []
    for row in _coqa_rows(split):
        if len(out) >= n:
            break
        out.append(
            Example(
                prompt=cfg.dataset.prompt_template.format(
                    story=row["story"], question=row["question"]
                ),
                # HalluShift uses `answer['text']` only, discarding the three
                # `additional_answers` (hal_detection.py:323).
                gold=[row["answer"]],
                idx=len(out),
            )
        )
    return out


LOADERS = {
    "triviaqa": lambda cfg, n, sp: load_triviaqa(cfg, n, split=sp),
    "truthfulqa": lambda cfg, n, sp: load_truthfulqa(cfg, n),
    "coqa": lambda cfg, n, sp: load_coqa(cfg, n, split=sp),
}

# Which upstream split each dataset's pool is extracted from. Every dataset
# here mirrors HalluShift EXACTLY (hal_detection.py:39-79): HalluShift loads a
# single upstream split per dataset and carves train/eval out of *that* via a
# stratified in-split split, rather than training on one corpus and testing on
# another -- so there is no separate `<name>_test` corpus; make_split() carves
# out the eval slice at train time instead.
SPLIT_SOURCES = {
    # HalluShift loads triviaqa's `validation` split (deduped), not `train`.
    "triviaqa": {"train": "validation"},
    # HalluShift always loads CoQA's dev file, never the train file.
    "coqa": {"train": "dev"},
    # TruthfulQA has exactly one split (817 rows, `validation`).
    "truthfulqa": {"train": "validation"},
}


def load_examples(cfg) -> list[Example]:
    name = cfg.dataset.name
    if name not in LOADERS:
        raise KeyError(
            f"no loader for dataset {name!r}. Known: {sorted(LOADERS)}"
        )
    split = SPLIT_SOURCES[name]["train"]
    return LOADERS[name](cfg, cfg.dataset.n_samples, split)
