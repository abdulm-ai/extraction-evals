"""Intervals and paired tests.

Fields on one document are correlated (a blurry scan hurts all of them), so
intervals resample whole documents (a clustered bootstrap, as recommended in
Miller, "Adding Error Bars to Evals", 2024). Two runs on the same documents
are compared with McNemar's exact test per field, with Holm's correction
across fields.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binomtest, norm


def bootstrap_ci(values: np.ndarray, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile interval of per-document values (0/1 or fractions).
    NaN entries (field excluded on that document) are dropped first."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def bootstrap_ratio_ci(num: np.ndarray, den: np.ndarray, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0):
    """Interval for sum(num)/sum(den), resampling documents. Used for line-item
    subfields, where each document contributes a different number of items."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    keep = den > 0
    num, den = num[keep], den[keep]
    if den.sum() == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-9)
    lo, hi = np.quantile(r, [alpha / 2, 1 - alpha / 2])
    return float(num.sum() / den.sum()), float(lo), float(hi)


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired exact test on per-document correctness of one field in runs A and B."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep].astype(bool), b[keep].astype(bool)
    n01 = int((a & ~b).sum())   # right in A, wrong in B
    n10 = int((~a & b).sum())   # wrong in A, right in B
    n = n01 + n10
    p = 1.0 if n == 0 else binomtest(n01, n, 0.5).pvalue
    return {"n": int(len(a)), "a": float(a.mean()) if len(a) else math.nan,
            "b": float(b.mean()) if len(b) else math.nan, "a_only": n01, "b_only": n10, "p": float(p)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def mcnemar_mde(n: int, discordant: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest accuracy difference a paired test on n documents detects with
    the given power, when a share `discordant` of documents disagree between
    the runs (Connor, 1987, normal approximation)."""
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    pd = max(discordant, 1e-6)
    lo, hi = 0.0, pd
    for _ in range(60):
        d = (lo + hi) / 2
        need = (za * math.sqrt(pd) + zb * math.sqrt(max(pd - d * d, 0))) ** 2 / (d * d)
        if need > n:
            lo = d
        else:
            hi = d
    return hi


def auroc(score: np.ndarray, is_error: np.ndarray) -> float:
    """Probability a wrong field gets a lower confidence than a right one."""
    s, e = np.asarray(score, float), np.asarray(is_error, bool)
    pos, neg = s[~e], s[e]   # correct fields should score higher
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(prob: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    p, c = np.asarray(prob, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if m.any():
            total += m.mean() * abs(p[m].mean() - c[m].mean())
    return float(total)


def risk_coverage(score: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Accept from the most confident down. Ties are accepted together, so a
    signal with few distinct values gets few points on the curve."""
    s, c = np.asarray(score, float), np.asarray(correct, float)
    order = np.argsort(-s, kind="stable")
    s, c = s[order], c[order]
    cov, risk = [], []
    n = len(s)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        k = j + 1
        cov.append(k / n)
        risk.append(1 - c[:k].mean())
        i = k
    return np.array(cov), np.array(risk)


def coverage_at_risk(score: np.ndarray, correct: np.ndarray, max_risk: float) -> float:
    cov, risk = risk_coverage(score, correct)
    ok = cov[risk <= max_risk]
    return float(ok.max()) if len(ok) else 0.0
