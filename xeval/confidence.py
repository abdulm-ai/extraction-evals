"""Confidence signals per field, and how well each one predicts an error.

Signals for a field of one document:
  stated       the model's own 0-1 rating, asked for in the output
  checks       0 if a failed domain check touches the field, 1 if only passing
               checks do, 0.5 if no check covers it
  consistency  share of the other samples (same model, same input) that give
               the same normalised value
  cross_model  1 if a run of another model gives the same normalised value
  logprob      probability of the value's tokens (OpenAI non-reasoning models
               only; OpenAI's reasoning models and the Gemini models used here
               return no log-probabilities)
  combined     logistic regression over the available signals, cross-fitted
               over test documents in 5 folds
"""

from __future__ import annotations

import json
import math
import re

import numpy as np

from .normalize import canon
from .results import DocResult, RunScores
from .schema import DocType

SIGNALS = ["stated", "checks", "consistency", "cross_model", "logprob", "combined"]


def value_key(doc: DocType, field: str, out: dict | None):
    """A hashable normalised value, for agreement between outputs."""
    if out is None:
        return ("<failed>",)
    if field == "line_items":
        items = out.get("line_items") or []
        return tuple(sorted(
            tuple(str(canon(doc, s.name, it.get(s.name), item=True)) for s in doc.item if s.name != "description")
            for it in items if isinstance(it, dict)))
    return str(canon(doc, field, out.get(field)))


def _check_signal(d: DocResult, field: str) -> float:
    if d.pred is None or d.score.correct(field) is None:
        return math.nan
    final = d.attempts[-1] if d.attempts else {}
    touched = [c for c in final.get("checks", []) if field in c["fields"]]
    if not touched:
        return 0.5
    return 0.0 if any(not c["ok"] for c in touched) else 1.0


_VAL = r'"{f}"\s*:\s*(null|-?[\d.eE+-]+|"(?:[^"\\]|\\.)*")'


def _logprob_signal(d: DocResult, field: str) -> float:
    a = d.attempts[-1] if d.attempts else None
    if not a or "logprobs" not in a or d.pred is None:
        return math.nan
    toks = a["logprobs"]
    text = "".join(t for t, _ in toks)
    starts = np.cumsum([0] + [len(t) for t, _ in toks])
    if field == "line_items":
        i = text.find('"line_items"')
        if i < 0:
            return math.nan
        depth, j0, j = 0, text.find("[", i), None
        for k in range(j0, len(text)):
            depth += text[k] == "["
            depth -= text[k] == "]"
            if depth == 0:
                j = k
                break
        if j is None:
            return math.nan
        span = (j0, j + 1)
        lps = [lp for (t, lp), s in zip(toks, starts[:-1]) if span[0] <= s < span[1]]
        return float(math.exp(min(lps))) if lps else math.nan
    m = re.search(_VAL.format(f=re.escape(field)), text)
    if not m:
        return math.nan
    span = m.span(1)
    lps = [lp for (t, lp), s, e in zip(toks, starts[:-1], starts[1:]) if s < span[1] and e > span[0]]
    return float(math.exp(sum(lps))) if lps else math.nan


def field_signals(rs: RunScores, other: RunScores | None = None) -> list[dict]:
    """One row per (document, field) with correctness and every signal."""
    other_by = {d.record["id"]: d for d in other.docs} if other else {}
    rows = []
    for d in rs.docs:
        doc = d.doc
        for f in doc.scored_fields:
            c = d.score.correct(f)
            if c is None:
                continue
            row = {"doc_id": d.record["id"], "set": d.record["set"], "field": f, "correct": float(c)}
            conf = (d.pred or {}).get("confidence") if isinstance((d.pred or {}).get("confidence"), dict) else None
            v = conf.get(f) if conf else None
            row["stated"] = float(min(max(v, 0.0), 1.0)) if isinstance(v, (int, float)) else math.nan
            row["checks"] = _check_signal(d, f)
            if len(d.samples) > 1 and d.pred is not None:
                mine = value_key(doc, f, d.pred)
                others = [value_key(doc, f, s) for s in d.samples[1:]]
                row["consistency"] = float(np.mean([o == mine for o in others]))
            else:
                row["consistency"] = math.nan
            od = other_by.get(d.record["id"])
            row["cross_model"] = (float(value_key(doc, f, od.pred) == value_key(doc, f, d.pred))
                                  if od is not None and d.pred is not None else math.nan)
            row["logprob"] = _logprob_signal(d, f)
            rows.append(row)
    return rows


def fit_combined(rows: list[dict], features: list[str]) -> np.ndarray:
    """Logistic regression (Newton's method, small L2), NaN features set to 0.5."""
    X = np.array([[1.0] + [0.5 if math.isnan(r[f]) else r[f] for f in features] for r in rows])
    y = np.array([r["correct"] for r in rows])
    w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + 1e-2 * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X + 1e-2 * np.eye(len(w))
        w -= np.linalg.solve(Hm, g)
    return w


def apply_combined(rows: list[dict], features: list[str], w: np.ndarray) -> None:
    for r in rows:
        x = np.array([1.0] + [0.5 if math.isnan(r[f]) else r[f] for f in features])
        r["combined"] = float(1 / (1 + math.exp(-x @ w)))


def cross_fit_combined(rows: list[dict], features: list[str], folds: int = 5) -> None:
    """Out-of-fold combined score: split by document, fit on the other folds,
    score the held-out one. No document is scored by a model that saw it."""
    import hashlib
    fold = {i: int(hashlib.sha256(i.encode()).hexdigest(), 16) % folds for i in {r["doc_id"] for r in rows}}
    for k in range(folds):
        w = fit_combined([r for r in rows if fold[r["doc_id"]] != k], features)
        apply_combined([r for r in rows if fold[r["doc_id"]] == k], features, w)


def doc_level(rows: list[dict], signal: str) -> tuple[np.ndarray, np.ndarray]:
    """Document confidence = its least confident field; correct = every field right."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["doc_id"], []).append(r)
    conf, corr = [], []
    for rs in by.values():
        vals = [r[signal] for r in rs]
        if any(math.isnan(v) for v in vals):
            continue
        conf.append(min(vals))
        corr.append(float(all(r["correct"] for r in rs)))
    return np.array(conf), np.array(corr)


def save(rows: list[dict], path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
