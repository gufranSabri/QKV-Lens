# Manual labeling: export one CSV per (dataset, llm) with a blank label column, then import the filled-in labels into every variant's manifest.

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from src.config import REPO_ROOT, Config
from src.extract.run_extraction import _atomic_write_text, parse_meta
from src.utils.logger import get_logger

logger = get_logger(__name__)

SHEETS_DIR = REPO_ROOT / "scripts" / "exports" / "label_sheets"
COLUMNS = ("idx", "gold", "llm_response", "label")
CHECK = "response_id"


def sheet_path(cfg: Config, sheets_dir: str | Path | None = None) -> Path:
    return Path(sheets_dir or SHEETS_DIR) / f"{cfg.dataset.name}_{cfg.llm.alias}.csv"


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def _response_id(text: str) -> str:
    # Letter prefix keeps Excel from reading it as a number or date.
    return "r" + hashlib.sha1(_norm(text).encode("utf-8")).hexdigest()[:10]


def _read_manifest(root: Path) -> list[dict]:
    with open(root / "manifest.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def _export(roots: list[Path], path: Path) -> int:
    rows: dict[int, tuple[str, str]] = {}
    for root in roots:
        for rec in _read_manifest(root):
            if rec["idx"] not in rows:
                meta = parse_meta(root / rec["dir"] / "meta.txt")
                rows[rec["idx"]] = (meta["gold"], meta["response"])

    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow((*COLUMNS, CHECK))
    for idx in sorted(rows):
        gold, response = rows[idx]
        w.writerow([idx, gold, response, "", _response_id(response)])
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: the BOM makes Excel open it as UTF-8 instead of the system codepage.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8-sig", newline="")
    tmp.replace(path)
    return len(rows)


def _read_sheet(path: Path) -> dict[int, tuple[int | None, str | None, str]]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:  # Excel's plain "CSV" save on Windows
        text = raw.decode("cp1252")
    header = text.splitlines()[0] if text else ""
    delimiter = max((",", ";", "\t"), key=header.count)  # some Excel locales save with ';'
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
    names = [h.strip().lower() for h in next(reader, [])]
    missing = [c for c in COLUMNS if c not in names]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}; expected {list(COLUMNS)}")
    col = {c: names.index(c) for c in (*COLUMNS, CHECK) if c in names}

    out: dict[int, tuple[int | None, str | None, str]] = {}
    errors = []
    for line_no, row in enumerate(reader, start=2):
        if not any(cell.strip() for cell in row):
            continue
        row = row + [""] * (len(names) - len(row))
        idx_cell, label_cell = row[col["idx"]].strip(), row[col["label"]].strip()
        try:
            idx = int(float(idx_cell))
        except ValueError:
            errors.append(f"line {line_no}: idx {idx_cell!r} is not a number")
            continue
        if label_cell in ("", "-1"):
            label = None
        elif label_cell in ("0", "1", "0.0", "1.0"):
            label = int(float(label_cell))
        else:
            errors.append(f"line {line_no} (idx {idx}): label {label_cell!r} must be 0, 1 or blank")
            continue
        if idx in out and out[idx][0] != label:
            errors.append(f"line {line_no}: idx {idx} appears twice with different labels")
            continue
        out[idx] = (label, row[col[CHECK]].strip() if CHECK in col else None, row[col["llm_response"]])

    if errors:
        shown = "\n  ".join(errors[:20])
        more = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"{path}: {len(errors)} bad row(s), nothing imported:\n  {shown}{more}")
    return out


def label_from_sheet(cfg: Config, sheets_dir: str | Path | None = None) -> int:
    # Returns how many examples are still unlabeled in at least one variant afterwards.
    pair = cfg.pair_dir(cfg.dataset.name, cfg.llm.alias)
    roots = sorted((p.parent for p in pair.glob("*/manifest.jsonl")), key=lambda r: (r.name != "default", r.name))
    if not roots:
        raise FileNotFoundError(f"no */manifest.jsonl under {pair}; run `extract` first")
    path = sheet_path(cfg, sheets_dir)

    if not path.exists():
        n = _export(roots, path)
        logger.info("exported %d row(s) to %s -- fill in the label column (1 = hallucinated) and rerun", n, path)
        return len({r["idx"] for root in roots for r in _read_manifest(root) if r.get("label") not in (0, 1)})

    sheet = _read_sheet(path)
    filled = {idx: v for idx, v in sheet.items() if v[0] is not None}
    logger.info("%s: %d row(s), %d labeled", path, len(sheet), len(filled))

    known: set[int] = set()
    remaining: set[int] = set()
    for root in roots:
        records = _read_manifest(root)
        known.update(r["idx"] for r in records)
        n_set = 0
        mismatched = []
        for rec in records:
            entry = filled.get(rec["idx"])
            # Rows already holding the sheet's label need no check; only changes are verified against meta.txt.
            if entry is None or rec.get("label") == entry[0]:
                continue
            label, check, response = entry
            stored = parse_meta(root / rec["dir"] / "meta.txt")["response"]
            if (_response_id(stored) != check) if check else (_norm(stored) != _norm(response)):
                mismatched.append(rec["idx"])
                continue
            rec["score"], rec["label"] = float(label), label
            n_set += 1
        if n_set:
            _atomic_write_text(
                root / "manifest.jsonl",
                "".join(json.dumps(r) + "\n" for r in sorted(records, key=lambda r: r["idx"])),
            )
        unlabeled = {r["idx"] for r in records if r.get("label") not in (0, 1)}
        remaining |= unlabeled
        logger.info("%-9s set %d label(s), %d unlabeled left", root.name, n_set, len(unlabeled))
        if mismatched:
            logger.warning(
                "%s: %d sheet row(s) skipped because the stored response changed since the "
                "export (re-extracted) or %s was edited, e.g. idx %s",
                root.name, len(mismatched), CHECK, mismatched[:10],
            )

    unknown = sorted(set(filled) - known)
    if unknown:
        logger.warning("%d sheet row(s) have an idx in no manifest, e.g. %s", len(unknown), unknown[:10])
    missing = len(known - set(sheet))
    if missing:
        logger.warning(
            "%d extracted example(s) are not in %s (extracted after the export); "
            "move the sheet away and rerun to export a fresh one",
            missing, path,
        )
    return len(remaining)
