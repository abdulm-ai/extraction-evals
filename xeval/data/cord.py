"""CORD v2 (Park et al., 2019; CC BY 4.0) to this repo's gold format.

CORD's own validation split is used as dev and its test split as test.
Labels are the strings printed on the receipt; they are parsed once here.
A label that can't be read as one value (two amounts in one field, a
percentage where an amount should be) excludes that field from scoring and is
recorded with the reason, instead of being guessed.
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

from PIL import Image

from ..normalize import Unparseable, parse_printed_amount, parse_printed_count

REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
FILES = {
    "dev": "data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
    "test": "data/test-00000-of-00001-9c204eb3f4e11791.parquet",
}
URL = "https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/{rev}/{path}"

HEADER_MAP = {
    "subtotal": ("sub_total", "subtotal_price"),
    "discount": ("sub_total", "discount_price"),
    "service_charge": ("sub_total", "service_price"),
    "tax": ("sub_total", "tax_price"),
    "total": ("total", "total_price"),
    "cash_tendered": ("total", "cashprice"),
    "change_due": ("total", "changeprice"),
}

MAX_SIDE = 1568  # long side sent to the models


def download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split, path in FILES.items():
        dst = raw_dir / f"{split}.parquet"
        if dst.exists():
            continue
        print(f"downloading CORD {split} ...")
        urllib.request.urlretrieve(URL.format(rev=REVISION, path=path), dst)


def _one_amount(v):
    """One amount, or a list of copies of the same amount (CORD labels both
    when a value is printed twice)."""
    if isinstance(v, str):
        return parse_printed_amount(v)
    vals = {parse_printed_amount(x) for x in v if isinstance(x, str)}
    if len(vals) != 1:
        raise Unparseable(f"several values: {v!r}")
    return vals.pop()


def _num(v) -> float:
    return float(v)


def convert_parse(p: dict) -> tuple[dict, dict[str, str]]:
    """gt_parse -> (gold, {excluded field: reason})."""
    gold: dict = {}
    excluded: dict[str, str] = {}
    for name, (sec, key) in HEADER_MAP.items():
        v = (p.get(sec) or {}).get(key) if isinstance(p.get(sec), dict) else None
        if v is None:
            gold[name] = None
            continue
        try:
            gold[name] = _num(abs(_one_amount(v)))
        except Unparseable as e:
            gold[name] = None
            excluded[name] = str(e)
    menu = p.get("menu") or []
    if isinstance(menu, dict):
        menu = [menu]
    items = []
    for it in menu:
        try:
            nm = it.get("nm")
            if isinstance(nm, list):
                raise Unparseable(f"item name split in parts: {nm!r}")
            price = it.get("price")
            if isinstance(price, list):
                raise Unparseable(f"several prices on one line: {price!r}")
            unit = it.get("unitprice")
            cnt = it.get("cnt")
            items.append({
                "description": nm,
                "quantity": _num(parse_printed_count(cnt)) if isinstance(cnt, str) else None,
                "unit_price": _num(abs(parse_printed_amount(unit))) if isinstance(unit, str) else None,
                "line_total": _num(abs(parse_printed_amount(price))) if isinstance(price, str) else None,
            })
        except Unparseable as e:
            excluded["line_items"] = str(e)
            break
    gold["line_items"] = items
    return gold, excluded


def ocr_text_from_words(valid_line: list[dict]) -> str:
    """CORD's word-level transcription, laid out in rows by position.

    It only covers annotated lines, so it is pre-filtered as well as perfect:
    an upper bound for OCR input, not a realistic one.
    """
    words = []
    for line in valid_line:
        for w in line.get("words", []):
            q = w["quad"]
            words.append(((q["y1"] + q["y3"]) / 2, q["x1"], q["y3"] - q["y1"], w["text"]))
    words.sort()
    rows: list[list[tuple]] = []
    for w in words:
        if rows and abs(w[0] - rows[-1][0][0]) < max(6, 0.5 * w[2]):
            rows[-1].append(w)
        else:
            rows.append([w])
    return "\n".join(" ".join(t for *_, t in sorted(r, key=lambda w: w[1])) for r in rows)


def build(raw_dir: Path, out_dir: Path, gold_dir: Path) -> None:
    import pyarrow.parquet as pq

    download(raw_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)
    for split in FILES:
        t = pq.read_table(raw_dir / f"{split}.parquet")
        img_dir = out_dir / split
        img_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for i, row in enumerate(t.to_pylist()):
            doc_id = f"cord-{split}-{i:03d}"
            gt = json.loads(row["ground_truth"])
            gold, excluded = convert_parse(gt["gt_parse"])
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            if max(img.size) > MAX_SIDE:
                img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
            img_path = img_dir / f"{i:03d}.jpg"
            img.save(img_path, quality=90)
            (img_dir / f"{i:03d}.gt.txt").write_text(ocr_text_from_words(gt["valid_line"]), encoding="utf-8")
            records.append({
                "id": doc_id, "set": "cord", "split": split, "doc_type": "receipt",
                "pages": [img_path.as_posix()],
                "gt_text": (img_dir / f"{i:03d}.gt.txt").as_posix(),
                "gold": gold, "excluded": excluded,
            })
        with open(gold_dir / f"cord_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_ex = sum(bool(r["excluded"]) for r in records)
        print(f"cord {split}: {len(records)} receipts, {n_ex} with an excluded field")
