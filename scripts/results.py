"""Build results/: results.json, tables.md and the charts.

Reads committed predictions only (runs/*/predictions.jsonl), so it runs
without an API key. python -m xeval report runs this file.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xeval import confidence as C  # noqa: E402
from xeval.diff import compare  # noqa: E402
from xeval.results import cost_latency, field_table, load_run, mean_field_accuracy  # noqa: E402
from xeval.run import RUNS  # noqa: E402
from xeval.stats import auroc, bootstrap_ci, coverage_at_risk, ece, risk_coverage  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

MAIN = ["luna-6", "terra-6.1", "sol-6.1", "gpt-4.1-mini-lp"]
LABEL = {"luna-6": "GPT-6 Luna", "terra-6.1": "GPT-6.1 Terra", "sol-6.1": "GPT-6.1 Sol", "gpt-4.1-mini-lp": "GPT-4.1 mini"}
INPUT_MODELS = ["luna-6", "sol-6.1"]
OUTPUT_MODELS = ["luna-6", "terra-6.1", "sol-6.1"]
COLORS = {"luna-6": "#7aa6d6", "terra-6.1": "#3b6ea5", "sol-6.1": "#1d3557", "gpt-4.1-mini-lp": "#c15f3c"}
MODES = ["image", "ocr", "image+ocr", "gt_text"]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 140})


def exists(name: str) -> bool:
    return (RUNS / name / "predictions.jsonl").exists()


def pct(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}"


def ci(r):
    return f"{pct(r['acc'])} ({pct(r['lo'])}-{pct(r['hi'])})"


results: dict = {"runs": {}}
md: list[str] = []

# --- per-run tables ----------------------------------------------------------
for name in sorted(p.name for p in RUNS.iterdir() if (p / "predictions.jsonl").exists()):
    rs = load_run(name)
    entry = {"config": rs.config, "sets": {}}
    for s in sorted({d.record["set"] for d in rs.docs}):
        sub = rs.by_set(s)
        entry["sets"][s] = {"fields": field_table(sub), "first_attempt": field_table(sub, first=True),
                            "cost": cost_latency(sub), "mean_field_acc": mean_field_accuracy(sub)}
    results["runs"][name] = entry

# --- main comparison: per field, per model -----------------------------------
main = [m for m in MAIN if exists(m)]
for s in ("cord", "invoices"):
    if not main:
        break
    fields = [r["field"] for r in results["runs"][main[0]]["sets"][s]["fields"]]
    md.append(f"\n### {s}: field accuracy, % (95% interval)\n")
    md.append("| Field | " + " | ".join(LABEL[m] for m in main) + " |")
    md.append("|---|" + "---|" * len(main))
    for f in fields:
        cells = []
        for m in main:
            row = next(r for r in results["runs"][m]["sets"][s]["fields"] if r["field"] == f)
            cells.append(ci(row))
        md.append(f"| {f} | " + " | ".join(cells) + " |")
    md.append("| mean of fields | " + " | ".join(pct(results["runs"][m]["sets"][s]["mean_field_acc"]) for m in main) + " |")
    md.append("| $ per 1,000 docs | " + " | ".join(f"{results['runs'][m]['sets'][s]['cost']['cost_per_1k']:.2f}" for m in main) + " |")
    md.append("| p50 / p95 s | " + " | ".join(f"{results['runs'][m]['sets'][s]['cost']['p50_s']:.1f} / {results['runs'][m]['sets'][s]['cost']['p95_s']:.1f}" for m in main) + " |")

# --- chart: all-fields-correct vs cost ---------------------------------------
if main:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, s in zip(axes, ("cord", "invoices")):
        for m in main:
            e = results["runs"][m]["sets"][s]
            allf = next(r for r in e["fields"] if r["field"] == "ALL FIELDS")
            x = e["cost"]["cost_per_1k"]
            color = COLORS[m]
            ax.errorbar([x], [100 * allf["acc"]], yerr=[[100 * (allf["acc"] - allf["lo"])], [100 * (allf["hi"] - allf["acc"])]],
                        fmt="o", color=color, capsize=3)
            ax.annotate(LABEL[m], (x, 100 * allf["acc"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("$ per 1,000 documents (log scale)")
        ax.set_ylabel("documents with every field right, %")
        ax.set_title({"cord": "CORD receipts (100, test)", "invoices": "Synthetic invoices (100, test)"}[s])
    fig.suptitle("Accuracy against cost (bars are 95% intervals)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_cost.png")
    plt.close(fig)

# --- input modes --------------------------------------------------------------
im_models = [m for m in INPUT_MODELS if exists(m)]
if im_models:
    md.append("\n### Input modes: documents with every field right, %\n")
    md.append("| Model | set | " + " | ".join(MODES) + " |")
    md.append("|---|---|" + "---|" * len(MODES))
    for m in im_models:
        for s in ("cord", "invoices"):
            cells = []
            for mode in MODES:
                n = m if mode == "image" else f"{m}-{mode}"
                if n in results["runs"]:
                    e = results["runs"][n]["sets"][s]
                    allf = next(r for r in e["fields"] if r["field"] == "ALL FIELDS")
                    cells.append(f"{pct(allf['acc'])} (${e['cost']['cost_per_1k']:.2f})")
                else:
                    cells.append("-")
            md.append(f"| {LABEL[m]} | {s} | " + " | ".join(cells) + " |")

# --- output modes -------------------------------------------------------------
om = [m for m in OUTPUT_MODELS if exists(m) and exists(f"{m}-free")]
if om:
    md.append("\n### Strict structured output vs free JSON (paired, test)\n")
    md.append("| Model | set | strict | free | failed to parse (free) | repaired (strict / free) |")
    md.append("|---|---|---|---|---|---|")
    for m in om:
        for s in ("cord", "invoices"):
            a, b = results["runs"][m]["sets"][s], results["runs"][f"{m}-free"]["sets"][s]
            fa = next(r for r in a["fields"] if r["field"] == "ALL FIELDS")
            fb = next(r for r in b["fields"] if r["field"] == "ALL FIELDS")
            fb_first = next(r for r in b["first_attempt"] if r["field"] == "ALL FIELDS")
            md.append(f"| {LABEL[m]} | {s} | {ci(fa)} | {ci(fb)} | {b['cost']['failed']} | {a['cost']['repaired']} / {b['cost']['repaired']} |")

# --- repair outcomes ----------------------------------------------------------
md.append("\n### What the repair turn did (sample 0, test, image input)\n")
md.append("| Model | repaired docs | field fixed | field broken | unchanged and still wrong | extra $ per 1,000 docs |")
md.append("|---|---|---|---|---|---|")
repair_rows = {}
for m in main:
    rs = load_run(m)
    fixed = broken = stuck = n = 0
    extra = 0.0
    for d in rs.docs:
        if len(d.attempts) < 2:
            continue
        n += 1
        extra += sum(a["cost"] for a in d.attempts[1:])
        for f in d.doc.scored_fields:
            a, b = d.first_score.correct(f), d.score.correct(f)
            if a is None or b is None:
                continue
            fixed += (not a) and b
            broken += a and (not b)
            stuck += (not a) and (not b) and C.value_key(d.doc, f, d.first) == C.value_key(d.doc, f, d.pred)
    repair_rows[m] = {"docs": n, "fixed": fixed, "broken": broken, "stuck": stuck, "extra_per_1k": 1000 * extra / len(rs.docs)}
    md.append(f"| {LABEL[m]} | {n} | {fixed} | {broken} | {stuck} | {1000 * extra / len(rs.docs):.2f} |")
results["repair"] = repair_rows

# --- confidence signals -------------------------------------------------------
# Cross-model agreement: a cheap model checked against a stronger one.
pairs = {"luna-6": "sol-6.1", "gpt-4.1-mini-lp": "sol-6.1", "terra-6.1": "sol-6.1", "sol-6.1": "terra-6.1"}
sig_res = {}
for m in [x for x in ("luna-6", "gpt-4.1-mini-lp", "terra-6.1", "sol-6.1") if exists(x)]:
    other = load_run(pairs[m]) if exists(pairs[m]) else None
    rows = C.field_signals(load_run(m), other)
    feats = [f for f in ("stated", "checks", "consistency", "cross_model", "logprob") if not all(math.isnan(r[f]) for r in rows)]
    dev_name = f"dev-{m}"
    if exists(dev_name):
        dev_other = load_run(f"dev-{pairs[m]}") if exists(f"dev-{pairs[m]}") else None
        dev_rows = C.field_signals(load_run(dev_name), dev_other)
        dfeats = [f for f in feats if not all(math.isnan(r[f]) for r in dev_rows)]
        w = C.fit_combined(dev_rows, dfeats)
        C.apply_combined(rows, dfeats, w)
        feats_all = feats + ["combined"]
    else:
        feats_all = feats
    C.save(rows, OUT / f"signals_{m}.jsonl")
    out = {}
    for f in feats_all:
        sc = np.array([r[f] for r in rows])
        corr = np.array([r["correct"] for r in rows])
        keep = ~np.isnan(sc)
        dconf, dcorr = C.doc_level(rows, f)
        out[f] = {
            "auroc": auroc(sc[keep], corr[keep] == 0),
            "ece": ece(sc[keep], corr[keep]) if f in ("stated", "consistency", "logprob", "combined") else math.nan,
            "field_cov_at_1pct": coverage_at_risk(sc[keep], corr[keep], 0.01),
            "doc_cov_at_2pct": coverage_at_risk(dconf, dcorr, 0.02) if len(dconf) else math.nan,
            "doc_cov_at_5pct": coverage_at_risk(dconf, dcorr, 0.05) if len(dconf) else math.nan,
            "field_error_rate": float(1 - corr[keep].mean()),
            "doc_error_rate": float(1 - dcorr.mean()) if len(dcorr) else math.nan,
        }
    sig_res[m] = out
results["signals"] = sig_res
if sig_res:
    md.append("\n### Which signals predict an error (test, every field of 200 documents)\n")
    md.append("| Model | signal | AUROC | ECE | fields auto-accepted at ≤1% field error | docs auto-accepted at ≤2% / ≤5% doc error |")
    md.append("|---|---|---|---|---|---|")
    for m, out in sig_res.items():
        for f, v in out.items():
            e = "-" if math.isnan(v["ece"]) else f"{v['ece']:.3f}"
            md.append(f"| {LABEL[m]} | {f} | {v['auroc']:.2f} | {e} | "
                      f"{pct(v['field_cov_at_1pct'])}% | {pct(v['doc_cov_at_2pct'])}% / {pct(v['doc_cov_at_5pct'])}% |")

    # risk-coverage chart, document level, one panel per model with signals
    ms = list(sig_res)
    fig, axes = plt.subplots(1, len(ms), figsize=(3.6 * len(ms), 3.6), squeeze=False)
    colors = {"stated": "#888888", "checks": "#7a5195", "consistency": "#ef5675", "cross_model": "#ffa600",
              "logprob": "#2f4b7c", "combined": "#003f5c"}
    for ax, m in zip(axes[0], ms):
        rows = [json.loads(l) for l in open(OUT / f"signals_{m}.jsonl", encoding="utf-8")]
        for f in sig_res[m]:
            dconf, dcorr = C.doc_level(rows, f)
            if not len(dconf):
                continue
            cov, risk = risk_coverage(dconf, dcorr)
            ax.step(100 * cov, 100 * risk, where="post", label=f, color=colors[f], lw=1.6)
        ax.set_title(LABEL[m], fontsize=10)
        ax.set_xlabel("documents auto-accepted, %")
        ax.set_ylim(0, None)
    axes[0][0].set_ylabel("accepted documents with an error, %")
    axes[0][-1].legend(fontsize=8, frameon=False)
    fig.suptitle("Risk against coverage: accept the most confident documents first", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "risk_coverage.png")
    plt.close(fig)

    # reliability diagram for stated confidence
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    for m in ms:
        rows = [json.loads(l) for l in open(OUT / f"signals_{m}.jsonl", encoding="utf-8")]
        p = np.array([r["stated"] for r in rows if not math.isnan(r["stated"])])
        c = np.array([r["correct"] for r in rows if not math.isnan(r["stated"])])
        if not len(p):
            continue
        edges = np.array([0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001])
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mk = (p >= lo) & (p < hi)
            if mk.sum() >= 5:
                xs.append(p[mk].mean())
                ys.append(c[mk].mean())
        ax.plot(xs, ys, "o-", label=LABEL[m], ms=4)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("share actually correct")
    ax.set_title("Stated confidence against accuracy", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png")
    plt.close(fig)

# --- degradation curve --------------------------------------------------------
deg = [m for m in OUTPUT_MODELS if exists(f"{m}-degrade")]
if deg:
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    curves = {}
    for m in deg:
        rs = load_run(f"{m}-degrade")
        ys, los, his = [], [], []
        for lv in range(6):
            v = np.array([float(d.score.all_correct) for d in rs.docs if d.record["meta"]["degrade"] == lv])
            mean, lo, hi = bootstrap_ci(v)
            ys.append(mean), los.append(lo), his.append(hi)
        curves[m] = ys
        ax.plot(range(6), 100 * np.array(ys), "o-", label=LABEL[m], ms=4)
        ax.fill_between(range(6), 100 * np.array(los), 100 * np.array(his), alpha=0.1)
    ax.set_xlabel("wear level (0 = clean render)")
    ax.set_ylabel("invoices with every field right, %")
    ax.set_title("Same 40 invoices, increasingly worn", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "degradation.png")
    plt.close(fig)
    results["degradation"] = curves

# --- regression demo ---------------------------------------------------------
reg = {}
for m in OUTPUT_MODELS:
    if exists(m) and exists(f"{m}-v2"):
        rows, summ = compare(m, f"{m}-v2")
        reg[m] = {"rows": rows, "summary": summ}
results["regression"] = reg

(OUT / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
(OUT / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"wrote {OUT / 'results.json'} and {OUT / 'tables.md'}")
