"""python -m xeval diff <run_a> <run_b> [--alpha 0.05] [--min-drop 0.0]

Per-field paired comparison of two runs on the same documents. Each field gets
McNemar's exact test on per-document correctness; p-values are Holm-adjusted
across fields. Exit status 1 if any field got significantly worse in B, which
is what a CI job keys on.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from .results import load_run, mean_field_accuracy
from .stats import holm, mcnemar, mcnemar_mde


def compare(a_name: str, b_name: str) -> tuple[list[dict], dict]:
    a, b = load_run(a_name), load_run(b_name)
    ids = sorted({d.record["id"] for d in a.docs} & {d.record["id"] for d in b.docs})
    da = {d.record["id"]: d for d in a.docs}
    db = {d.record["id"]: d for d in b.docs}
    fields: list[str] = []
    for i in ids:
        for f in da[i].doc.scored_fields:
            if f not in fields:
                fields.append(f)
    rows = []
    for f in fields + ["ALL FIELDS"]:
        va, vb = [], []
        for i in ids:
            if f == "ALL FIELDS":
                va.append(float(da[i].score.all_correct))
                vb.append(float(db[i].score.all_correct))
                continue
            x, y = da[i].score.correct(f), db[i].score.correct(f)
            ok = f in da[i].doc.scored_fields and x is not None and y is not None
            va.append(float(x) if ok else np.nan)
            vb.append(float(y) if ok else np.nan)
        t = mcnemar(np.array(va), np.array(vb))
        t["field"] = f
        t["mde"] = mcnemar_mde(t["n"], max((t["a_only"] + t["b_only"]) / max(t["n"], 1), 0.02))
        rows.append(t)
    adj = holm({r["field"]: r["p"] for r in rows if r["field"] != "ALL FIELDS"})
    for r in rows:
        r["p_holm"] = adj.get(r["field"], r["p"])
    summary = {"a": a_name, "b": b_name, "n_docs": len(ids),
               "mean_a": mean_field_accuracy(a), "mean_b": mean_field_accuracy(b)}
    return rows, summary


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="xeval diff")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-drop", type=float, default=0.0, help="ignore significant drops smaller than this")
    args = ap.parse_args(argv)
    rows, s = compare(args.a, args.b)
    print(f"A = {s['a']}\nB = {s['b']}\n{s['n_docs']} documents in both\n")
    print(f"{'field':<16} {'A':>6} {'B':>6} {'change':>7}  {'A only':>6} {'B only':>6}  {'p (Holm)':>9}  {'detectable':>10}")
    regressions = []
    for r in rows:
        ch = r["b"] - r["a"]
        flag = ""
        if r["field"] != "ALL FIELDS" and ch < 0 and r["p_holm"] < args.alpha and -ch >= args.min_drop:
            flag = "  REGRESSION"
            regressions.append(r["field"])
        elif r["field"] != "ALL FIELDS" and ch > 0 and r["p_holm"] < args.alpha:
            flag = "  better"
        p = r["p_holm"] if r["field"] != "ALL FIELDS" else r["p"]
        print(f"{r['field']:<16} {100 * r['a']:6.1f} {100 * r['b']:6.1f} {100 * ch:+7.1f}  {r['a_only']:>6} {r['b_only']:>6}  "
              f"{p:9.4f}  {100 * r['mde']:9.1f}%{flag}")
    print(f"\nmean field accuracy: A {100 * s['mean_a']:.1f}, B {100 * s['mean_b']:.1f} ({100 * (s['mean_b'] - s['mean_a']):+.1f})")
    print("'detectable': the smallest drop this many documents would catch with 80% power at this disagreement rate.")
    if regressions:
        print(f"\nFAIL: significant regression in {', '.join(regressions)}")
        return 1
    print("\nOK: no field got significantly worse")
    return 0
