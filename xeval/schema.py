"""Field specs for the two document types.

One spec per document type, used three ways: to build the JSON schema the model
is asked for, to validate what comes back, and as the scorer's field list.
"""

from __future__ import annotations

from dataclasses import dataclass

# kinds: money, number, text, id, date, currency


@dataclass(frozen=True)
class Field:
    name: str
    kind: str
    hint: str


@dataclass(frozen=True)
class DocType:
    name: str
    header: tuple[Field, ...]
    item: tuple[Field, ...]
    # Money fields where a printed 0 means the same as the field being absent.
    zero_is_null: frozenset[str] = frozenset()

    @property
    def header_names(self) -> list[str]:
        return [f.name for f in self.header]

    @property
    def scored_fields(self) -> list[str]:
        """Header fields plus `line_items` as one composite field."""
        return self.header_names + ["line_items"]

    def field(self, name: str) -> Field:
        for f in self.header + self.item:
            if f.name == name:
                return f
        raise KeyError(name)


ITEM = (
    Field("description", "text", "the item's name as printed"),
    Field("quantity", "number", "units, hours or weight; null if not printed"),
    Field("unit_price", "money", "price of one unit; null if not printed"),
    Field("line_total", "money", "the amount charged for the line"),
)

RECEIPT = DocType(
    name="receipt",
    header=(
        Field("subtotal", "money", "sum of the items before tax, service and discounts"),
        Field("discount", "money", "total discount, as a positive amount"),
        Field("service_charge", "money", "service charge"),
        Field("tax", "money", "tax amount"),
        Field("total", "money", "the amount to pay"),
        Field("cash_tendered", "money", "cash handed over by the customer"),
        Field("change_due", "money", "change given back"),
    ),
    item=ITEM,
    zero_is_null=frozenset({"discount", "service_charge", "tax", "change_due"}),
)

INVOICE = DocType(
    name="invoice",
    header=(
        Field("vendor_name", "text", "the company that issued the invoice, not the customer"),
        Field("invoice_number", "id", "the invoice's own number"),
        Field("issue_date", "date", "date the invoice was issued, YYYY-MM-DD"),
        Field("due_date", "date", "date payment is due, YYYY-MM-DD"),
        Field("currency", "currency", "ISO 4217 code, e.g. EUR"),
        Field("subtotal", "money", "sum of the line totals before discount and tax"),
        Field("discount", "money", "total discount, as a positive amount"),
        Field("shipping", "money", "shipping, freight or delivery charge"),
        Field("tax", "money", "total tax (VAT, GST, sales tax), all rates added up"),
        Field("amount_paid", "money", "payments or deposits already received and deducted on the invoice"),
        Field("total", "money", "the amount still to pay: the balance due if a payment is deducted"),
    ),
    item=ITEM,
    zero_is_null=frozenset({"discount", "shipping", "tax", "amount_paid"}),
)

DOC_TYPES = {"receipt": RECEIPT, "invoice": INVOICE}

_JSON_TYPE = {
    "money": "number",
    "number": "number",
    "text": "string",
    "id": "string",
    "date": "string",
    "currency": "string",
}


def json_schema(doc: DocType, with_confidence: bool = True) -> dict:
    """A strict-compatible JSON schema: every key required, nulls explicit."""

    def prop(f: Field) -> dict:
        return {"type": [_JSON_TYPE[f.kind], "null"], "description": f.hint}

    item = {
        "type": "object",
        "properties": {f.name: prop(f) for f in doc.item},
        "required": [f.name for f in doc.item],
        "additionalProperties": False,
    }
    props = {f.name: prop(f) for f in doc.header}
    props["line_items"] = {"type": "array", "items": item}
    required = doc.header_names + ["line_items"]
    if with_confidence:
        props["confidence"] = {
            "type": "object",
            "description": "for each field, the probability (0 to 1) that your value is correct",
            "properties": {n: {"type": "number"} for n in doc.scored_fields},
            "required": doc.scored_fields,
            "additionalProperties": False,
        }
        required.append("confidence")
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }
