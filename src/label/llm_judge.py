# Correctness labeling via an LLM judge, replacing BLEURT: Novita AI's
# OpenAI-compatible API (llm_judge) or a local Ollama server (ollama_judge).
# label = 1 if the judge says the response hallucinates.

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from contextlib import nullcontext
from functools import lru_cache, partial
from pathlib import Path

from src.label.exact_match import _as_list

NOVITA_BASE_URL = "https://api.novita.ai/openai"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
REQUEST_TIMEOUT = 30
# Judge prompts are a few hundred tokens; a fixed context keeps Ollama from sizing it to the GPU.
OLLAMA_NUM_CTX = 4096
OLLAMA_LOAD_TIMEOUT = 900

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


def _novita_chat(model_name: str, messages: list[dict]) -> str:
    response = _get_client(model_name).chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=100,
        extra_body={"enable_thinking": False},
    )
    return response.choices[0].message.content or ""


def _ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    return host if host.startswith("http") else f"http://{host}"


def _ollama_post(path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        _ollama_host() + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _ollama_chat(model_name: str, messages: list[dict]) -> str:
    out = _ollama_post("/api/chat", {
        "model": model_name, "messages": messages, "stream": False, "think": False, "keep_alive": "60m",
        "options": {"temperature": 0, "num_predict": 16, "num_ctx": OLLAMA_NUM_CTX},
    }, REQUEST_TIMEOUT)
    return out["message"]["content"] or ""


def _ollama_ready(model_name: str) -> None:
    # Fails loudly before labeling: an unreachable server would otherwise label everything 0.
    host = _ollama_host()
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=10) as resp:
            names = {m["name"] for m in json.loads(resp.read()).get("models", [])}
    except OSError as e:
        raise RuntimeError(f"no Ollama server at {host} ({e}); start it with `bash scripts/ollama.sh serve`") from e
    if model_name not in names and f"{model_name}:latest" not in names:
        raise RuntimeError(f"{model_name} is not pulled on {host} (has {sorted(names)}); "
                           f"run `bash scripts/ollama.sh pull {model_name}`")
    from src.utils.logger import get_logger

    get_logger(__name__).info("Ollama: loading %s onto the GPU (can take a few minutes)", model_name)
    _ollama_post("/api/generate", {"model": model_name, "keep_alive": "60m",
                                   "options": {"num_ctx": OLLAMA_NUM_CTX}}, OLLAMA_LOAD_TIMEOUT)


def _do_request(chat, messages, result_box):
    # Runs in a worker thread so a hard wall-clock deadline can be enforced
    # even if the client's own timeout never raises.
    try:
        result_box["content"] = chat(messages)
    except Exception as e:  # noqa: BLE001 - surfaced to the caller below
        result_box["error"] = e


def _query_judge(chat, answer: str, refs: list[str]) -> int | None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_input(answer, refs)},
    ]

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        result_box: dict = {}
        worker = threading.Thread(
            target=_do_request, args=(chat, messages, result_box), daemon=True
        )
        worker.start()
        worker.join(timeout=REQUEST_TIMEOUT)

        if worker.is_alive():
            last_err = f"request hard-hung past {REQUEST_TIMEOUT}s"
        elif "error" in result_box:
            last_err = str(result_box["error"])
        else:
            content = result_box["content"].strip()
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

    label = _query_judge(partial(_novita_chat, model_name), answer, refs)
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
    backend: str = "novita",
) -> list[tuple[float, int]]:
    from src.utils.logger import get_logger
    from src.utils.progress import progress

    _log = get_logger(__name__)
    cache = _load_cache(cache_path)
    _log.info(
        "LLM judge: scoring %d examples with %s model=%s (%d judgments cached at %s)",
        len(answers), backend, model_name, len(cache), cache_path,
    )
    if backend == "ollama":
        _ollama_ready(model_name)
        chat, cache_model = partial(_ollama_chat, model_name), f"ollama/{model_name}"
    else:
        chat, cache_model = partial(_novita_chat, model_name), model_name

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
            key = _cache_key(cache_model, answer, refs)
            label = cache.get(key)
            if label is not None:
                n_hits += 1
            else:
                n_queried += 1
                label = _query_judge(chat, answer, refs)
                if label is not None and cache_f is not None:
                    cache_f.write(json.dumps({"key": key, "label": label}) + "\n")
                    cache_f.flush()
                    cache[key] = label
            if label is None:
                label = 0
            out.append((float(label), label))

    _log.info("LLM judge: %d cache hits, %d queried", n_hits, n_queried)
    return out
