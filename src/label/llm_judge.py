# Correctness labeling via a cloud LLM judge (Novita AI's OpenAI-compatible
# API), replacing BLEURT. label = 1 if the judge says the response hallucinates.

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

from src.label.exact_match import _as_list

NOVITA_BASE_URL = "https://api.novita.ai/openai"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
REQUEST_TIMEOUT = 30

SYSTEM_PROMPT = (
    "You are a strict hallucination judge. You will be given a model's "
    "generated response and a list of gold (reference) answers. Decide "
    "whether the generated response hallucinates, i.e. it does NOT match "
    "/ is NOT supported by / contradicts or omits the information in the "
    "gold answers. If the generated response agrees with (or is a "
    "reasonable paraphrase / subset / superset that is consistent with) "
    "any one of the gold answers, it is NOT a hallucination.\n\n"
    "Respond with exactly one character and nothing else:\n"
    "1 = hallucination (response does not match the gold answers)\n"
    "0 = not a hallucination (response matches/is supported by a gold answer)\n"
    "Do not output any explanation, punctuation, or whitespace other than "
    "the single digit."
)


def _load_dotenv(path):
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


@lru_cache(maxsize=1)
def _get_client(model_name: str):
    from openai import OpenAI

    _load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", ".env"))

    api_key = os.environ.get("NOVITA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NOVITA_API_KEY not set. Add it to scripts/.env or export it in your shell."
        )
    return OpenAI(api_key=api_key, base_url=NOVITA_BASE_URL)


def _build_user_input(answer: str, refs: list[str]) -> str:
    gold_block = "\n".join(f"- {g}" for g in refs)
    return (
        f"Gold answers:\n{gold_block}\n\n"
        f"Generated response:\n{answer}\n\n"
        "Does the generated response hallucinate relative to the gold "
        "answers? Answer with 1 or 0 only."
    )


def _extract_binary_label(text: str) -> int | None:
    for ch in text:
        if ch == "1":
            return 1
        if ch == "0":
            return 0
    return None


def _do_request(client, model_name, messages, result_box):
    # Runs in a worker thread so a hard wall-clock deadline can be enforced
    # even if the SDK's own timeout never raises.
    try:
        result_box["response"] = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=100,
            extra_body={"enable_thinking": False},
        )
    except Exception as e:  # noqa: BLE001 - surfaced to the caller below
        result_box["error"] = e


def _query_judge(client, model_name: str, answer: str, refs: list[str]) -> int | None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_input(answer, refs)},
    ]

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        result_box: dict = {}
        worker = threading.Thread(
            target=_do_request, args=(client, model_name, messages, result_box), daemon=True
        )
        worker.start()
        worker.join(timeout=REQUEST_TIMEOUT)

        if worker.is_alive():
            last_err = f"request hard-hung past {REQUEST_TIMEOUT}s"
        elif "error" in result_box:
            last_err = str(result_box["error"])
        else:
            content = (result_box["response"].choices[0].message.content or "").strip()
            label = _extract_binary_label(content)
            if label is not None:
                return label
            last_err = f"unparseable response: content={content!r}"

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    print(f"    WARNING: giving up after {MAX_RETRIES} attempts ({last_err}); labeling as 0", file=sys.stderr)
    return None


def score_llm_judge(
    answer: str,
    gold,
    model_name: str = "deepseek/deepseek-v4-pro",
) -> tuple[float, int]:
    refs = _as_list(gold)
    if not refs:
        return 0.0, 1

    client = _get_client(model_name)
    label = _query_judge(client, model_name, answer, refs)
    if label is None:
        label = 0
    return float(label), label


def _cache_key(model_name: str, answer: str, refs: list[str]) -> str:
    payload = json.dumps([model_name, SYSTEM_PROMPT, answer, refs])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache(path: Path | None) -> dict[str, int]:
    cache: dict[str, int] = {}
    if path is None or not path.exists():
        return cache
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # truncated line from a crash mid-write
                continue
            cache[rec["key"]] = rec["label"]
    return cache


def score_llm_judge_batch(
    answers: list[str],
    golds: list,
    model_name: str = "deepseek/deepseek-v4-pro",
    cache_path: Path | None = None,
) -> list[tuple[float, int]]:
    from src.utils.logger import get_logger
    from src.utils.progress import progress

    _log = get_logger(__name__)
    cache = _load_cache(cache_path)
    _log.info(
        "LLM judge: scoring %d examples with model=%s (%d judgments cached at %s)",
        len(answers), model_name, len(cache), cache_path,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    out = []
    n_hits = n_queried = 0
    with open(cache_path, "a", encoding="utf-8") if cache_path is not None else nullcontext() as cache_f:
        for answer, gold in progress(list(zip(answers, golds)), desc="LLM judge scoring", ncols=100):
            refs = _as_list(gold)
            if not refs:
                out.append((0.0, 1))
                continue
            key = _cache_key(model_name, answer, refs)
            label = cache.get(key)
            if label is not None:
                n_hits += 1
            else:
                n_queried += 1
                label = _query_judge(_get_client(model_name), model_name, answer, refs)
                if label is not None and cache_f is not None:
                    cache_f.write(json.dumps({"key": key, "label": label}) + "\n")
                    cache_f.flush()
                    cache[key] = label
            if label is None:
                label = 0
            out.append((float(label), label))

    _log.info("LLM judge: %d cache hits, %d queried", n_hits, n_queried)
    return out
