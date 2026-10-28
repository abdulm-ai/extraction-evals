"""Normalisation before comparison.

Predictions arrive as JSON (numbers as numbers, dates as YYYY-MM-DD). Gold
labels from CORD arrive as the strings printed on the receipt ("16,500",
"Rp 9.736", "@ 12.000"), so they get parsed here once, when the gold is built.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from decimal import Decimal, InvalidOperation

from .schema import DocType


class Unparseable(ValueError):
    """A gold string that can't be read as one value. The field is excluded
    from scoring and listed in the gold audit instead of being guessed."""


_NUM_CHARS = re.compile(r"[^0-9.,\-]")


def parse_printed_amount(s: str) -> Decimal:
    """Parse an amount as printed on an Indonesian receipt.

    Both `.` and `,` are used as thousands separators ("16,500", "23.000").
    A final separator followed by exactly two digits is a decimal point
    ("70000.00"); followed by three digits it groups thousands.
    """
    if s is None:
        raise Unparseable("empty")
    raw = s.strip()
    if raw in {"-", "--", ""}:
        return Decimal("0")  # printed as a dash: nothing charged
    if raw.startswith("[") or raw.count("Rp") > 1:
        raise Unparseable(f"several values: {s!r}")
    if "%" in raw:
        raise Unparseable(f"a percentage, not an amount: {s!r}")
    t = _NUM_CHARS.sub("", raw)
    neg = t.startswith("-") or raw.rstrip().endswith("-")
    t = t.replace("-", "")
    t = t.strip(".,")
    if not t or not any(ch.isdigit() for ch in t):
        raise Unparseable(f"no digits: {s!r}")
    seps = [i for i, ch in enumerate(t) if ch in ".,"]
    if seps:
        last = seps[-1]
        tail = t[last + 1 :]
        if len(tail) == 2:
            intpart = re.sub(r"[.,]", "", t[:last])
            t = f"{intpart}.{tail}"
        elif len(tail) == 3:
            groups = re.split(r"[.,]", t)
            if not all(len(g) == 3 for g in groups[1:]) or not (1 <= len(groups[0]) <= 3):
                raise Unparseable(f"irregular digit groups: {s!r}")
            t = "".join(groups)
        else:
            raise Unparseable(f"ambiguous separator: {s!r}")
    try:
        v = Decimal(t)
    except InvalidOperation as e:
        raise Unparseable(s) from e
    return -v if neg else v


def parse_printed_count(s: str) -> Decimal:
    """Quantities: "1", "1X", "2 x", "1.00", "4.00xITEMs"."""
    m = re.match(r"[^\d]{0,3}?(\d+(?:[.,]\d+)?)", (s or "").strip())
    if not m:
        raise Unparseable(f"no count: {s!r}")
    return Decimal(m.group(1).replace(",", "."))


# --- canonical forms used by the scorer -------------------------------------

def money(v) -> Decimal | None:
    if v is None:
        return None
    if isinstance(v, bool):
        raise TypeError("bool is not an amount")
    return Decimal(str(v)).copy_abs().quantize(Decimal("0.01"))


def number(v) -> Decimal | None:
    if v is None:
        return None
    return Decimal(str(v)).quantize(Decimal("0.001"))


def text(v) -> str | None:
    """Case-folded, accents kept, punctuation treated as space."""
    if v is None:
        return None
    s = unicodedata.normalize("NFKC", str(v)).casefold()
    s = re.sub(r"[^\w]+", " ", s)
    s = " ".join(s.split())
    return s or None


def ident(v) -> str | None:
    if v is None:
        return None
    s = re.sub(r"\s+", "", str(v)).upper()
    return s or None


_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def date(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if not _DATE.match(s):
        return f"invalid:{s}"
    try:
        dt.date.fromisoformat(s)
    except ValueError:
        return f"invalid:{s}"
    return s


def currency(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().upper()
    return s or None


CANON = {
    "money": money,
    "number": number,
    "text": text,
    "id": ident,
    "date": date,
    "currency": currency,
}


def canon(doc: DocType, name: str, v, item: bool = False):
    """Canonical value of one field, with the zero-means-absent rule applied."""
    f = doc.field(name)
    try:
        c = CANON[f.kind](v)
    except (TypeError, InvalidOperation, ValueError):
        return f"invalid:{v!r}"
    if not item and name in doc.zero_is_null and c == Decimal("0.00"):
        return None
    return c


def anls(a: str | None, b: str | None, threshold: float = 0.5) -> float:
    """Normalised Levenshtein similarity, cut to 0 below the threshold (as in DocVQA)."""
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    a, b = text(a) or "", text(b) or ""
    if not a and not b:
        return 1.0
    d = levenshtein(a, b)
    s = 1.0 - d / max(len(a), len(b))
    return s if s >= threshold else 0.0


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
