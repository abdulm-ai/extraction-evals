"""Per-field scoring.

Header fields: exact match after normalisation, per document. A field the gold
marks as excluded (unparseable label) is skipped, never guessed.

Line items: matched to gold items as an assignment problem (Hungarian
algorithm), then scored per subfield. Matching by position would count one
skipped line as every later line being wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .normalize import anls, canon
from .schema import DocType

MATCH_MIN = 0.3  # below this similarity a pair is two unrelated items
DESC_MIN = 0.8   # ANLS at or above which an item description counts as right


@dataclass
class FieldScore:
    correct: bool
    gold: object
    pred: object
    hallucinated: bool = False  # gold is null, prediction is not
    missed: bool = False        # gold has a value, prediction is null
    anls: float | None = None


@dataclass
class ItemScores:
    n_gold: int
    n_pred: int
    n_matched: int
    n_correct: int                       # matched and every scored subfield right
    sub_correct: dict[str, int] = field(default_factory=dict)
    sub_total: dict[str, int] = field(default_factory=dict)
    desc_anls: list[float] = field(default_factory=list)

    @property
    def all_correct(self) -> bool:
        return self.n_gold == self.n_pred == self.n_correct


@dataclass
class DocScore:
    doc_id: str
    fields: dict[str, FieldScore]
    items: ItemScores

    def correct(self, name: str) -> bool | None:
        if name == "line_items":
            return self.items.all_correct
        fs = self.fields.get(name)
        return None if fs is None else fs.correct

    @property
    def all_correct(self) -> bool:
        return all(f.correct for f in self.fields.values()) and self.items.all_correct


def _value(doc: DocType, name: str, v, item: bool = False):
    return canon(doc, name, v, item=item)


def score_header(doc: DocType, gold: dict, pred: dict, excluded: set[str]) -> dict[str, FieldScore]:
    out = {}
    for f in doc.header:
        if f.name in excluded:
            continue
        g = _value(doc, f.name, gold.get(f.name))
        p = _value(doc, f.name, pred.get(f.name)) if pred is not None else None
        fs = FieldScore(correct=(g == p), gold=g, pred=p)
        fs.hallucinated = g is None and p is not None
        fs.missed = g is not None and p is None
        if f.kind in ("text", "id"):
            fs.anls = anls(gold.get(f.name), pred.get(f.name) if pred else None)
        out[f.name] = fs
    return out


def _item_sim(doc: DocType, g: dict, p: dict) -> float:
    s = 0.6 * anls(g.get("description"), p.get("description"))
    gl, pl = _value(doc, "line_total", g.get("line_total"), True), _value(doc, "line_total", p.get("line_total"), True)
    if gl is not None and gl == pl:
        s += 0.4
    return s


def score_items(doc: DocType, gold_items: list[dict], pred_items: list[dict]) -> ItemScores:
    subs = [f.name for f in doc.item]
    sc = ItemScores(len(gold_items), len(pred_items), 0, 0,
                    {s: 0 for s in subs}, {s: 0 for s in subs})
    # Subfields are scored only where the gold has a value: CORD annotates
    # quantity and unit price on some lines and not others.
    for g in gold_items:
        for s in subs:
            if g.get(s) is not None:
                sc.sub_total[s] += 1
    if not gold_items or not pred_items:
        return sc
    sim = np.array([[_item_sim(doc, g, p) for p in pred_items] for g in gold_items])
    rows, cols = linear_sum_assignment(-sim)
    for r, c in zip(rows, cols):
        if sim[r, c] < MATCH_MIN:
            continue
        g, p = gold_items[r], pred_items[c]
        sc.n_matched += 1
        sc.desc_anls.append(anls(g.get("description"), p.get("description")))
        ok_all = True
        for s in subs:
            if g.get(s) is None:
                continue
            if s == "description":
                # Printed item names are abbreviated and misspelt; a reading
                # within 80% similarity counts. Amounts must match exactly.
                ok = anls(g.get(s), p.get(s)) >= DESC_MIN
            else:
                ok = _value(doc, s, g.get(s), True) == _value(doc, s, p.get(s), True)
            sc.sub_correct[s] += ok
            ok_all &= ok
        sc.n_correct += ok_all
    return sc


def score_doc(doc: DocType, doc_id: str, gold: dict, pred: dict | None, excluded: set[str] | None = None) -> DocScore:
    excluded = excluded or set()
    if pred is None:
        # No valid output at all: every field counts as wrong, including
        # fields whose gold is null.
        fields = score_header(doc, gold, {}, excluded)
        for fs in fields.values():
            fs.correct = False
        items = score_items(doc, gold.get("line_items") or [], [])
        items.n_pred = -1  # never equal to n_gold, so line_items is wrong too
        return DocScore(doc_id, fields, items)
    fields = score_header(doc, gold, pred, excluded)
    items = score_items(doc, gold.get("line_items") or [], pred.get("line_items") or [])
    return DocScore(doc_id, fields, items)
