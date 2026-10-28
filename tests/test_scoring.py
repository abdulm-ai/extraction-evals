import copy
import json
from pathlib import Path

import numpy as np
import pytest

from xeval import stats
from xeval.checks import failed, run_checks
from xeval.normalize import Unparseable, anls, parse_printed_amount, parse_printed_count
from xeval.schema import INVOICE, RECEIPT, json_schema
from xeval.score import score_doc

ROOT = Path(__file__).resolve().parents[1]


def gold(name):
    path = ROOT / "gold" / name
    if not path.exists():
        pytest.skip(f"{name} not built (python -m xeval data)")
    return [json.loads(l) for l in path.open(encoding="utf-8")]


@pytest.mark.parametrize("s,v", [
    ("16,500", 16500), ("23.000", 23000), ("Rp 9,736", 9736), ("@ 12.000", 12000),
    ("70000.00", 70000), ("-9,545", -9545), (":9,000", 9000), ("1,555,200", 1555200), ("-", 0),
])
def test_printed_amounts(s, v):
    assert parse_printed_amount(s) == v


@pytest.mark.parametrize("s", ["Discount(0%)", "['1', '2']", "12,5", "abc"])
def test_unparseable_amounts(s):
    with pytest.raises(Unparseable):
        parse_printed_amount(s)


@pytest.mark.parametrize("s,v", [("1", 1), ("1X", 1), ("2 x", 2), ("1.00", 1), ("*2", 2), ("4.00xITEMs", 4)])
def test_printed_counts(s, v):
    assert parse_printed_count(s) == v


def test_gold_scores_perfect_against_itself():
    for name, doc in (("cord_dev.jsonl", RECEIPT), ("invoices_dev.jsonl", INVOICE)):
        for r in gold(name):
            s = score_doc(doc, r["id"], r["gold"], r["gold"], set(r["excluded"]))
            assert s.all_correct, r["id"]


def test_known_perturbations():
    r = gold("invoices_dev.jsonl")[0]
    g = r["gold"]
    p = copy.deepcopy(g)
    p["total"] = g["total"] + 1
    p["vendor_name"] = g["vendor_name"].upper() + "."   # case and punctuation are normalised away
    p["line_items"] = list(reversed(p["line_items"]))  # order doesn't matter
    s = score_doc(INVOICE, r["id"], g, p)
    assert not s.fields["total"].correct
    assert s.fields["vendor_name"].correct
    assert s.items.all_correct
    # dropping one item: the others still match
    p2 = copy.deepcopy(g)
    p2["line_items"] = p2["line_items"][1:]
    s2 = score_doc(INVOICE, r["id"], g, p2)
    assert s2.items.n_correct == len(g["line_items"]) - 1 and not s2.items.all_correct


def test_null_handling():
    g = {"subtotal": None, "discount": None, "service_charge": None, "tax": None, "total": 100.0,
         "cash_tendered": None, "change_due": None, "line_items": []}
    p = dict(g, discount=0, cash_tendered=100.0)
    s = score_doc(RECEIPT, "x", g, p)
    assert s.fields["discount"].correct          # a printed 0 discount is the same as none
    assert s.fields["cash_tendered"].hallucinated
    assert not score_doc(RECEIPT, "x", g, None).fields["subtotal"].correct  # failed run: all wrong


def test_checks_name_fields():
    g = {"vendor_name": "A", "invoice_number": "1", "issue_date": "2026-03-01", "due_date": "2026-02-01",
         "currency": "EUR", "subtotal": 10.0, "discount": None, "tax": 1.9, "total": 12.0,
         "line_items": [{"description": "x", "quantity": 2, "unit_price": 5.0, "line_total": 10.0}]}
    f = {c.name: c for c in failed(run_checks(INVOICE, g))}
    assert set(f) == {"total_formula", "due_after_issue"}
    assert "total" in f["total_formula"].fields


def test_schema_is_strict():
    s = json_schema(INVOICE)
    assert s["additionalProperties"] is False
    assert set(s["required"]) == set(s["properties"])


def test_anls():
    assert anls("EGG TART", "egg tart") == 1.0
    assert 0.8 < anls("PIZZA TOAST", "PIZA TOAST") < 1.0
    assert anls("abc", "xyz") == 0.0


def test_stats():
    rng = np.random.default_rng(1)
    v = rng.random(200) < 0.8
    m, lo, hi = stats.bootstrap_ci(v)
    assert lo < m < hi and abs(m - v.mean()) < 1e-12
    a = np.ones(100)
    b = a.copy()
    b[:12] = 0
    t = stats.mcnemar(a, b)
    assert t["a_only"] == 12 and t["p"] < 0.001
    adj = stats.holm({"x": 0.01, "y": 0.04, "z": 0.5})
    assert adj["x"] == pytest.approx(0.03) and adj["y"] == pytest.approx(0.08)
    score = np.array([0.9, 0.8, 0.7, 0.2])
    corr = np.array([1, 1, 1, 0])
    assert stats.auroc(score, ~corr.astype(bool)) == 1.0
    assert stats.coverage_at_risk(score, corr, 0.0) == 0.75
    assert 0 < stats.mcnemar_mde(100, 0.1) < 0.1
