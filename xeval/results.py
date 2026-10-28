"""Load runs and score them. Everything downstream (tables, charts, diff,
confidence signals) starts from `RunScores`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .run import RUNS, load_records
from .schema import DOC_TYPES
from .score import DocScore, score_doc

ROOT = Path(__file__).resolve().parents[1]


def load_corrections() -> dict[str, dict]:
    """gold/corrections.jsonl: {"id", "field", "value", "reason"} per line.
    `value` replaces the gold value; "exclude": true removes the field."""
    path = ROOT / "gold" / "corrections.jsonl"
    out: dict[str, dict] = {}
    if path.exists():
        for l in path.open(encoding="utf-8"):
            if l.strip():
                c = json.loads(l)
                out.setdefault(c["id"], {})[c["field"]] = c
    return out


def apply_corrections(rec: dict, corr: dict) -> dict:
    c = corr.get(rec["id"])
    if not c:
        return rec
    rec = json.loads(json.dumps(rec))
    for f, entry in c.items():
        if entry.get("exclude"):
            rec["excluded"][f] = "gold audit: " + entry["reason"]
        else:
            rec["gold"][f] = entry["value"]
    return rec


@dataclass
class DocResult:
    record: dict
    pred: dict | None            # final parsed output of sample 0
    first: dict | None           # first attempt of sample 0
    samples: list[dict | None]   # first attempt of every sample
    score: DocScore
    first_score: DocScore
    attempts: list[dict]
    cost: float
    latency: float

    @property
    def doc(self):
        return DOC_TYPES[self.record["doc_type"]]


@dataclass
class RunScores:
    name: str
    config: dict
    docs: list[DocResult]

    def by_set(self, s: str) -> "RunScores":
        return RunScores(self.name, self.config, [d for d in self.docs if d.record["set"] == s])

    def correct_matrix(self, field: str, first: bool = False) -> np.ndarray:
        """1/0 per document, NaN where the field doesn't apply or is excluded."""
        out = []
        for d in self.docs:
            sc = d.first_score if first else d.score
            v = sc.correct(field)
            out.append(np.nan if v is None or field not in d.doc.scored_fields else float(v))
        return np.array(out)


@lru_cache(maxsize=None)
def _gold(sets: tuple[str, ...], split: str, corrected: bool) -> dict[str, dict]:
    recs = load_records(list(sets), split)
    if corrected:
        corr = load_corrections()
        recs = [apply_corrections(r, corr) for r in recs]
    return {r["id"]: r for r in recs}


def load_run(name: str, corrected: bool = True) -> RunScores:
    d = RUNS / name
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    gold = _gold(tuple(cfg["sets"]), cfg["split"], corrected)
    by_doc: dict[str, dict[int, dict]] = {}
    for l in (d / "predictions.jsonl").open(encoding="utf-8"):
        p = json.loads(l)
        by_doc.setdefault(p["doc_id"], {})[p["sample"]] = p
    docs = []
    for doc_id, samples in by_doc.items():
        rec = gold[doc_id]
        dt = DOC_TYPES[rec["doc_type"]]
        s0 = samples[0]
        att = s0["attempts"]
        pred = att[s0["final"]]["parsed"] if s0["final"] is not None else None
        first = att[0]["parsed"] if att and att[0].get("parsed") is not None else None
        ex = set(rec["excluded"])
        docs.append(DocResult(
            record=rec, pred=pred, first=first,
            samples=[(samples[k]["attempts"][0].get("parsed") if samples[k]["attempts"] else None) for k in sorted(samples)],
            score=score_doc(dt, doc_id, rec["gold"], pred, ex),
            first_score=score_doc(dt, doc_id, rec["gold"], first, ex),
            attempts=att,
            cost=sum(a["cost"] for p in samples.values() for a in p["attempts"] if p["sample"] == 0),
            latency=sum(a["latency_s"] for a in att),
        ))
    docs.sort(key=lambda r: r.record["id"])
    return RunScores(name, cfg, docs)


def field_table(rs: RunScores, first: bool = False) -> list[dict]:
    """Per-field accuracy with 95% intervals, plus line-item subfields."""
    from .stats import bootstrap_ci, bootstrap_ratio_ci

    rows = []
    fields: list[str] = []
    for d in rs.docs:
        for f in d.doc.scored_fields:
            if f not in fields:
                fields.append(f)
    for f in fields:
        m, lo, hi = bootstrap_ci(rs.correct_matrix(f, first))
        n = int(np.sum(~np.isnan(rs.correct_matrix(f, first))))
        halluc = [d.score.fields[f].hallucinated for d in rs.docs if f in d.score.fields and d.score.fields[f].gold is None]
        rows.append({"field": f, "acc": m, "lo": lo, "hi": hi, "n": n,
                     "halluc": float(np.mean(halluc)) if halluc else np.nan, "n_null": len(halluc)})
    for sub in ("description", "quantity", "unit_price", "line_total"):
        num = [(d.first_score if first else d.score).items.sub_correct.get(sub, 0) for d in rs.docs]
        den = [(d.first_score if first else d.score).items.sub_total.get(sub, 0) for d in rs.docs]
        m, lo, hi = bootstrap_ratio_ci(np.array(num), np.array(den))
        rows.append({"field": f"item.{sub}", "acc": m, "lo": lo, "hi": hi, "n": int(sum(den)), "halluc": np.nan, "n_null": 0})
    allc = np.array([float((d.first_score if first else d.score).all_correct) for d in rs.docs])
    m, lo, hi = bootstrap_ci(allc)
    rows.append({"field": "ALL FIELDS", "acc": m, "lo": lo, "hi": hi, "n": len(allc), "halluc": np.nan, "n_null": 0})
    return rows


def mean_field_accuracy(rs: RunScores) -> float:
    """The single number this project argues against, computed for contrast:
    the unweighted mean over scored fields of their accuracy."""
    accs = [r["acc"] for r in field_table(rs) if not r["field"].startswith(("item.", "ALL"))]
    return float(np.nanmean(accs))


def cost_latency(rs: RunScores) -> dict:
    costs = np.array([d.cost for d in rs.docs])
    lat = np.array([d.latency for d in rs.docs])
    return {"cost_per_1k": float(costs.mean() * 1000), "p50_s": float(np.percentile(lat, 50)),
            "p95_s": float(np.percentile(lat, 95)),
            "failed": int(sum(d.pred is None for d in rs.docs)),
            "repaired": int(sum(len(d.attempts) > 1 for d in rs.docs))}
