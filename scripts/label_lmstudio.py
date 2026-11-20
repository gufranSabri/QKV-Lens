#!/usr/bin/env python3
# Standalone LM Studio judge for the label sheets from `detector.py label --mode export`; standard library only.
# Same prompt, gold parsing, answer parsing, retries and failure handling as src/label/llm_judge.py (keep in sync).
#
#   python3 label_lmstudio.py --model <lm-studio-model-id> triviaqa_llama3.1_8b.csv [more.csv ...]
#
# Fills the sheet's label column in place (1 = hallucinated) and can be stopped and rerun: labeled rows are skipped.
# Copy the filled sheets back to scripts/exports/label_sheets/ and run `detector.py label --mode export` to import.

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request

COLUMNS = ("idx", "gold", "llm_response", "label")
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
REQUEST_TIMEOUT = 30
MAX_TOKENS = 16
LOAD_TIMEOUT = 900
SAVE_EVERY = 25

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


class ThinkingOn(RuntimeError):
    pass


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed]
            except (ValueError, SyntaxError):
                pass
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


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


def _messages(answer: str, refs: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_input(answer, refs)},
    ]


def _chat(base_url: str, model: str, messages: list[dict], timeout: float) -> str:
    body = {
        "model": model, "messages": messages, "temperature": 0, "max_tokens": MAX_TOKENS, "stream": False,
        "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
    message = out["choices"][0]["message"]
    content = message.get("content") or ""
    # The cluster judge runs with thinking off; any reasoning here would make the labels a different protocol.
    if message.get("reasoning") or message.get("reasoning_content") or "<think>" in content:
        raise ThinkingOn(f"the model is thinking (reply {content[:80]!r}, reasoning "
                         f"{str(message.get('reasoning') or message.get('reasoning_content'))[:80]!r})")
    return content


def _do_request(base_url, model, messages, result_box):
    try:
        result_box["content"] = _chat(base_url, model, messages, REQUEST_TIMEOUT)
    except Exception as e:  # noqa: BLE001 - surfaced to the caller below
        result_box["error"] = e


def _query_judge(base_url: str, model: str, answer: str, refs: list[str]) -> int | None:
    messages = _messages(answer, refs)
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        result_box: dict = {}
        worker = threading.Thread(target=_do_request, args=(base_url, model, messages, result_box), daemon=True)
        worker.start()
        worker.join(timeout=REQUEST_TIMEOUT)

        if worker.is_alive():
            last_err = f"request hard-hung past {REQUEST_TIMEOUT}s"
        elif "error" in result_box:
            if isinstance(result_box["error"], ThinkingOn):
                raise result_box["error"]
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


def check_server(base_url: str, model: str) -> None:
    # Fails loudly before labeling: a dead server, a missing model or thinking left on would otherwise label everything 0.
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as resp:
            ids = [m["id"] for m in json.loads(resp.read()).get("data", [])]
    except OSError as e:
        raise SystemExit(f"no LM Studio server at {base_url} ({e}); start it in LM Studio (Developer -> Start Server)")
    if model not in ids:
        raise SystemExit(f"model {model!r} not on the server; pass --model as one of: {ids}")

    print(f"checking {model} (first request loads the model, can take minutes)...", flush=True)
    probes = ((["Paris"], "Paris", 0), (["Paris"], "Berlin", 1))
    for refs, answer, want in probes:
        try:
            content = _chat(base_url, model, _messages(answer, refs), LOAD_TIMEOUT).strip()
        except ThinkingOn as e:
            raise SystemExit(
                f"{e}.\nTurn thinking off for this model in LM Studio (model settings -> Thinking / Reasoning off, "
                "or add `{%- set enable_thinking = false %}` as the first line of its Prompt Template) and rerun."
            )
        except (OSError, RuntimeError) as e:
            raise SystemExit(f"test request failed: {e}")
        got = _extract_binary_label(content)
        if got != want:
            raise SystemExit(f"test request: gold {refs}, response {answer!r} -> reply {content!r}, expected {want}. "
                             "Check the model and its prompt template before labeling.")
    print("check passed: thinking is off and the model answers 0 / 1 correctly", flush=True)


def read_sheet(path: str) -> tuple[list[str], list[list[str]], str, dict[str, int]]:
    raw = open(path, "rb").read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    first = text.splitlines()[0] if text else ""
    delimiter = max((",", ";", "\t"), key=first.count)
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
    header, body = rows[0], rows[1:]
    names = [h.strip().lower() for h in header]
    missing = [c for c in COLUMNS if c not in names]
    if missing:
        raise SystemExit(f"{path}: missing column(s) {missing}; expected {list(COLUMNS)}")
    col = {c: names.index(c) for c in COLUMNS}
    body = [r + [""] * (len(header) - len(r)) for r in body]
    return header, body, delimiter, col


def write_sheet(path: str, header: list[str], body: list[list[str]], delimiter: str) -> None:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, delimiter=delimiter)
    w.writerow(header)
    w.writerows(body)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        f.write(buf.getvalue())
    os.replace(tmp, path)


def label_sheet(path: str, base_url: str, model: str, overwrite: bool) -> None:
    header, body, delimiter, col = read_sheet(path)
    labeled = lambda r: r[col["label"]].strip() in ("0", "1", "0.0", "1.0")
    rows = [r for r in body if any(c.strip() for c in r)]

    # Same unit of work as the cluster judge: one judgment per distinct (response, gold), shared by every row with it.
    groups: dict[tuple[str, str], list[list[str]]] = {}
    for r in rows:
        if overwrite or not labeled(r):
            groups.setdefault((r[col["llm_response"]], r[col["gold"]]), []).append(r)
    print(f"{path}: {len(rows)} rows, {len(rows) - sum(map(len, groups.values()))} already labeled, "
          f"{len(groups)} distinct (response, gold) to judge", flush=True)

    n_failed = done = 0
    start = time.time()
    try:
        for (answer, gold), members in groups.items():
            refs = _as_list(gold)
            if not refs:
                label = 1
            else:
                label = _query_judge(base_url, model, answer, refs)
                if label is None:
                    n_failed += 1
                    label = 0
            for r in members:
                r[col["label"]] = str(label)
            done += 1
            if done % SAVE_EVERY == 0:
                write_sheet(path, header, body, delimiter)
                rate = done / (time.time() - start)
                print(f"  {done}/{len(groups)} judged, {rate:.2f}/s, ~{(len(groups) - done) / rate / 60:.0f} min left",
                      flush=True)
    finally:
        write_sheet(path, header, body, delimiter)

    hall = sum(r[col["label"]].strip() in ("1", "1.0") for r in rows)
    print(f"{path}: done, {hall}/{len(rows)} hallucinated ({100 * hall / max(len(rows), 1):.1f}%)"
          + (f", {n_failed} judge failure(s) labeled 0" if n_failed else ""), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Label QKV-Lens label sheets with an LLM judge served by LM Studio.")
    ap.add_argument("sheets", nargs="+", help="label sheet CSV(s) from `detector.py label --mode export`")
    ap.add_argument("--model", required=True, help="LM Studio model identifier, e.g. qwen3.8-27b")
    ap.add_argument("--base-url", default="http://localhost:1234/v1", help="LM Studio's OpenAI-compatible server")
    ap.add_argument("--overwrite", action="store_true", help="relabel rows that already have a label")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(130))
    check_server(args.base_url, args.model)
    for path in args.sheets:
        try:
            label_sheet(path, args.base_url, args.model, args.overwrite)
        except ThinkingOn as e:
            raise SystemExit(f"{e}; stopped, progress saved. Turn thinking off in LM Studio and rerun.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nstopped; progress saved, rerun the same command to continue")
