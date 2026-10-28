"""Layer 1: does the reply parse, and does it match the schema?

With strict structured output this should never fail; with free JSON it
sometimes does, and the failure is named the same way a domain check is.
"""

from __future__ import annotations

import json
import re

from .schema import DocType

_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.S)


def parse_json(text: str) -> tuple[dict | None, str | None]:
    t = (text or "").strip()
    m = _FENCE.search(t)
    if m:
        t = m.group(1)
    elif not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i < 0 or j < i:
            return None, "the reply contains no JSON object"
        t = t[i : j + 1]
    try:
        v = json.loads(t)
    except json.JSONDecodeError as e:
        return None, f"the reply is not valid JSON ({e.msg} at character {e.pos})"
    if not isinstance(v, dict):
        return None, "the reply is JSON but not an object"
    return v, None


def _type_ok(kind: str, v) -> bool:
    if v is None:
        return True
    if kind in ("money", "number"):
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return isinstance(v, str)


def schema_errors(doc: DocType, out: dict) -> list[str]:
    errs = []
    for f in doc.header:
        if f.name not in out:
            errs.append(f"missing key {f.name}")
        elif not _type_ok(f.kind, out[f.name]):
            errs.append(f"{f.name} should be a {'number' if f.kind in ('money', 'number') else 'string'} or null, got {out[f.name]!r}")
    items = out.get("line_items")
    if not isinstance(items, list):
        errs.append("line_items should be a list")
    else:
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                errs.append(f"line_items[{i}] should be an object")
                continue
            for f in doc.item:
                if f.name not in it:
                    errs.append(f"line_items[{i}] is missing {f.name}")
                elif not _type_ok(f.kind, it[f.name]):
                    errs.append(f"line_items[{i}].{f.name} should be a number or null, got {it[f.name]!r}")
    return errs
