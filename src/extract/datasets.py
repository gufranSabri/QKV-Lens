# Text dataset loading: prompt construction and gold-answer extraction.
# Prompt templates follow HalluShift/ACT-ViT so responses are comparable.

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

    # TriviaQA repeats questions across contexts; HalluShift dedupes by question_id.
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
        # best_answer only -- HalluShift discards correct_answers (hal_detection.py:316)
        out.append(
            Example(
                prompt=cfg.dataset.prompt_template.format(question=row["question"]),
                gold=[row["best_answer"]],
                idx=len(out),
            )
        )
    return out


def _coqa_rows(split: str) -> list[dict]:
    # Download CoQA and flatten each dialogue turn into its own row. The
    # story accumulates preceding Q/A pairs (HalluShift hal_detection.py:81-128).
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
            story += f' Q: {question["input_text"]} A: {answer}'
            if story and story[-1] != ".":
                story += "."
    return rows


def load_coqa(cfg, n: int, split: str = "dev") -> list[Example]:
    out = []
    for row in _coqa_rows(split):
        if len(out) >= n:
            break
        out.append(
            Example(
                prompt=cfg.dataset.prompt_template.format(
                    story=row["story"], question=row["question"]
                ),
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

# Each dataset mirrors HalluShift exactly: one upstream split, train/eval
# carved out of it via make_split() rather than a separate test corpus.
SPLIT_SOURCES = {
    "triviaqa": {"train": "validation"},
    "coqa": {"train": "dev"},
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
