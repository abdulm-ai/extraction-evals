"""List gold audit candidates: fields where every model run agrees on one
value and that value differs from the gold label.

python scripts/audit.py <split> run [run ...]

Each candidate is then checked by looking at the document. Confirmed label
errors go into gold/corrections.jsonl with the reason; the results are
reported on both the raw and the corrected gold.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xeval.confidence import value_key  # noqa: E402
from xeval.results import load_run  # noqa: E402


def main(split: str, names: list[str]) -> None:
    runs = [load_run(n, corrected=False) for n in names]
    by = [{d.record["id"]: d for d in r.docs} for r in runs]
    ids = sorted(set.intersection(*(set(b) for b in by)))
    out = []
    for i in ids:
        ds = [b[i] for b in by]
        rec = ds[0].record
        doc = ds[0].doc
        for f in doc.scored_fields:
            if f in rec["excluded"] or any(d.pred is None for d in ds):
                continue
            vals = {value_key(doc, f, d.pred) for d in ds}
            if len(vals) == 1 and all(d.score.correct(f) is False for d in ds):
                pv = ds[0].pred.get(f) if f != "line_items" else [
                    (it.get("description"), it.get("quantity"), it.get("line_total")) for it in ds[0].pred.get("line_items") or []]
                gv = rec["gold"].get(f) if f != "line_items" else [
                    (it.get("description"), it.get("quantity"), it.get("line_total")) for it in rec["gold"].get("line_items") or []]
                out.append((i, f, gv, pv, rec["pages"][0]))
    print(f"{len(out)} candidates where all {len(names)} runs agree against the gold ({split})")
    for i, f, g, p, page in out:
        print(f"\n{i}  {f}  {page}\n  gold: {g}\n  runs: {p}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
