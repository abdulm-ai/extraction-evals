import json
import math

from xeval.confidence import _logprob_signal, value_key
from xeval.schema import INVOICE, RECEIPT
from xeval.validate import parse_json, schema_errors


def test_value_key_normalises():
    a = {"total": 1234.5, "line_items": [{"description": "x", "quantity": 1, "unit_price": 2, "line_total": 2}]}
    b = {"total": 1234.50, "line_items": [{"description": "X!", "quantity": 1.0, "unit_price": 2.0, "line_total": 2.0}]}
    assert value_key(RECEIPT, "total", a) == value_key(RECEIPT, "total", b)
    assert value_key(RECEIPT, "line_items", a) == value_key(RECEIPT, "line_items", b)
    assert value_key(RECEIPT, "discount", {"discount": 0}) == value_key(RECEIPT, "discount", {"discount": None})


class _Doc:
    def __init__(self, text, lps):
        toks = []
        for i, ch in enumerate(text):
            toks.append((ch, lps.get(i, 0.0)))
        self.attempts = [{"logprobs": toks}]
        self.pred = json.loads(text)


def test_logprob_signal_reads_the_value_tokens():
    text = '{"subtotal": 10, "total": 12, "line_items": [], "confidence": {"total": 0.9}}'
    i = text.index("12")
    d = _Doc(text, {i: math.log(0.5), i + 1: math.log(0.8)})
    assert abs(_logprob_signal(d, "total") - 0.4) < 1e-9
    assert _logprob_signal(d, "subtotal") == 1.0


def test_parse_and_schema_errors():
    v, err = parse_json('Here you go:\n```json\n{"total": "12"}\n```')
    assert err is None and v == {"total": "12"}
    errs = schema_errors(INVOICE, v)
    assert any("total should be a number" in e for e in errs)
    assert any("missing key vendor_name" in e for e in errs)
    assert parse_json("no json here")[0] is None
