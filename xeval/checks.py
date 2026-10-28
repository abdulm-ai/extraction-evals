"""Domain checks: the second validation layer, after the schema.

A check only runs when every value it needs is present, and it names the
fields it touches, so a failure can count against those fields as a
confidence signal and be named in a repair turn.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal

from .normalize import money, number
from .schema import DocType

ISO_4217 = set(
    "AED AUD BRL CAD CHF CNY CZK DKK EUR GBP HKD HUF IDR ILS INR JPY KRW MXN MYR NOK NZD "
    "PHP PLN SAR SEK SGD THB TRY TWD USD ZAR".split()
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    fields: tuple[str, ...]
    message: str


def _tol(doc: DocType, x: Decimal) -> Decimal:
    # Receipts here are in rupiah, rounded to the nearest 100 now and then;
    # invoices are in currencies with cents. Tolerances were set on the dev
    # split's gold labels, not on model output.
    if doc.name == "receipt":
        return max(Decimal("100"), abs(x) * Decimal("0.002"))
    return max(Decimal("0.02"), abs(x) * Decimal("0.0005"))


def _m(d: dict, k: str) -> Decimal | None:
    try:
        return money(d.get(k))
    except Exception:
        return None


def _zero(v: Decimal | None) -> Decimal:
    return v if v is not None else Decimal("0")


def run_checks(doc: DocType, out: dict) -> list[CheckResult]:
    res: list[CheckResult] = []
    items = out.get("line_items") or []
    sub, total = _m(out, "subtotal"), _m(out, "total")
    disc, tax = _m(out, "discount"), _m(out, "tax")
    svc = _m(out, "service_charge") if doc.name == "receipt" else _m(out, "shipping")
    paid = _m(out, "amount_paid") if doc.name == "invoice" else None

    # quantity x unit price = line total
    for i, it in enumerate(items):
        q = number(it.get("quantity")) if isinstance(it.get("quantity"), (int, float)) else None
        u, lt = _m(it, "unit_price"), _m(it, "line_total")
        if q is not None and u is not None and lt is not None and q != 0:
            exp = (q * u).quantize(Decimal("0.01"))
            ok = abs(exp - lt) <= _tol(doc, lt)
            res.append(CheckResult(
                "item_arithmetic", ok, ("line_items",),
                "" if ok else f"line {i + 1}: quantity {q} x unit price {u} = {exp}, but line_total is {lt}",
            ))

    line_totals = [_m(it, "line_total") for it in items]
    if items and all(v is not None for v in line_totals):
        s = sum(line_totals, Decimal("0"))
        if sub is not None:
            ok = abs(s - sub) <= _tol(doc, sub)
            # Line prices that already include tax and service, with the
            # subtotal printed net of them.
            if not ok and (tax is not None or svc is not None):
                ok = abs(s - (sub + _zero(tax) + _zero(svc))) <= _tol(doc, sub)
            res.append(CheckResult(
                "items_sum_to_subtotal", ok, ("line_items", "subtotal"),
                "" if ok else f"line totals add up to {s}, but subtotal is {sub}",
            ))
        elif total is not None and disc is None and tax is None and svc is None and paid is None:
            ok = abs(s - total) <= _tol(doc, total)
            res.append(CheckResult(
                "items_sum_to_total", ok, ("line_items", "total"),
                "" if ok else f"line totals add up to {s}, but total is {total} and there is no subtotal, tax, service or discount",
            ))

    if sub is not None and total is not None:
        exp = sub - _zero(disc) + _zero(tax) + _zero(svc) - _zero(paid)
        ok = abs(exp - total) <= _tol(doc, total)
        # On receipts, tax is often printed but already included in the subtotal.
        if not ok and doc.name == "receipt" and tax is not None:
            ok = abs((sub - _zero(disc) + _zero(svc)) - total) <= _tol(doc, total)
        extra = "service_charge" if doc.name == "receipt" else "shipping"
        fields = tuple(f for f, v in (("subtotal", sub), ("discount", disc), ("tax", tax), (extra, svc),
                                      ("amount_paid", paid), ("total", total)) if v is not None)
        res.append(CheckResult(
            "total_formula", ok, fields,
            "" if ok else f"subtotal {sub} - discount {_zero(disc)} + tax {_zero(tax)}"
            + (f" + service {_zero(svc)}" if doc.name == "receipt" else f" + shipping {_zero(svc)} - amount_paid {_zero(paid)}")
            + f" = {exp}, but total is {total}",
        ))

    if doc.name == "receipt":
        cash, change = _m(out, "cash_tendered"), _m(out, "change_due")
        if cash is not None and total is not None:
            exp = cash - total
            got = _zero(change)
            ok = abs(exp - got) <= _tol(doc, total) or (change is None and exp <= 0)
            res.append(CheckResult(
                "change_formula", ok, ("cash_tendered", "total", "change_due"),
                "" if ok else f"cash {cash} - total {total} = {exp}, but change_due is {change}",
            ))

    if doc.name == "invoice":
        issue, due = out.get("issue_date"), out.get("due_date")
        for name in ("issue_date", "due_date"):
            v = out.get(name)
            if v is not None:
                try:
                    dt.date.fromisoformat(str(v))
                    ok = len(str(v)) == 10
                except ValueError:
                    ok = False
                res.append(CheckResult(f"{name}_valid", ok, (name,), "" if ok else f"{name} {v!r} is not a real YYYY-MM-DD date"))
        try:
            if issue and due and dt.date.fromisoformat(due) < dt.date.fromisoformat(issue):
                res.append(CheckResult("due_after_issue", False, ("issue_date", "due_date"), f"due_date {due} is before issue_date {issue}"))
            elif issue and due:
                res.append(CheckResult("due_after_issue", True, ("issue_date", "due_date"), ""))
        except ValueError:
            pass
        cur = out.get("currency")
        if cur is not None:
            ok = str(cur).upper() in ISO_4217
            res.append(CheckResult("currency_code", ok, ("currency",), "" if ok else f"currency {cur!r} is not an ISO 4217 code"))
    return res


def failed(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if not r.ok]
