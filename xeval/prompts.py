"""Prompts, versioned. A run records the version it used.

v1 is the baseline. v2 is the "cleanup" used to demonstrate the regression
diff: shorter and tidier, with the due-date rule for payment terms dropped.
It was written to look like a harmless edit, which is the point.
"""

from __future__ import annotations

import json

from .schema import DocType, json_schema

RECEIPT_V1 = """You extract structured data from photos of shop and restaurant receipts.

Rules:
- Amounts are plain JSON numbers in the receipt's currency units. These receipts are from Indonesia and priced in rupiah, where "16,500" and "16.500" both mean sixteen thousand five hundred (16500).
- Use null for anything that isn't printed. Don't compute a value the receipt doesn't show, except as stated below.
- line_items: one entry per purchased item, in printed order. description is the item's own name exactly as printed, spelling and abbreviations included, without the modifiers printed under it (sizes, toppings, sugar level); modifiers are not separate lines either. line_total is the amount printed on the item's line.
- discount is a positive amount. A discount, tax or service charge printed as 0 can be given as 0 or null.
- cash_tendered is the cash the customer paid; change_due is the change returned. Card or e-wallet payments are not cash.
- confidence: for every field, the probability that your value is exactly right. Be honest: use low values where the print is unclear or the receipt is ambiguous."""

INVOICE_V1 = """You extract structured data from scanned business invoices.

Rules:
- vendor_name is the full name of the company that issued the invoice (its legal name, which may appear only in the footer), not the customer it is billed to.
- Dates are YYYY-MM-DD. Read the date format from the invoice's country and language: 05/03/2026 is 5 March on a British or French invoice and May 3 on an American one.
- due_date: if a due date is printed, use it. If only payment terms are given ("Net 30", "Zahlbar innerhalb von 30 Tagen", "Paiement à 30 jours"), due_date is issue_date plus that many days. If neither is given, null.
- currency is the ISO 4217 code (EUR, GBP, USD, CHF, CAD, AUD, INR, JPY ...), worked out from the symbol or code printed.
- Amounts are plain JSON numbers. Read thousands and decimal separators from the invoice's locale: 1.234,56 and 1 234,56 are one thousand two hundred thirty-four and 56 hundredths; 1,23,456.78 is Indian grouping.
- discount is a positive amount; null if there is none. shipping is any shipping, freight or delivery charge; null if none.
- tax is the total tax amount: add up every tax line (two VAT rates, or CGST plus SGST); null if no tax is charged.
- amount_paid is a payment or deposit already received and deducted on the invoice; null if none. total is the amount still to pay: when a payment is deducted, the balance due.
- line_items: every billed line, across all pages, in order.
- confidence: for every field, the probability that your value is exactly right. Be honest: use low values where the print is unclear or the invoice is ambiguous."""

INVOICE_V2 = """You extract structured data from invoices.

- vendor_name: the issuing company's full name (not the customer).
- Dates as YYYY-MM-DD, interpreted in the invoice's locale.
- due_date: the payment due date shown on the invoice, or null.
- currency: ISO 4217 code.
- Amounts as JSON numbers, with the invoice's own separators interpreted correctly.
- discount positive, or null. shipping: freight or delivery charge, or null.
- tax: all tax lines added up, or null. amount_paid: payments already deducted, or null.
- total: the amount still to pay.
- line_items: all billed lines, all pages, in order.
- confidence: per field, the probability your value is exactly right."""

SYSTEM = {
    ("receipt", "v1"): RECEIPT_V1,
    ("receipt", "v2"): RECEIPT_V1,
    ("invoice", "v1"): INVOICE_V1,
    ("invoice", "v2"): INVOICE_V2,
}

REPAIR = """Your output failed these checks:
{problems}

Look at the document again and reply with the complete corrected JSON. Change only values the document supports. If a check fails because the document itself doesn't add up, keep what is printed."""


def system_prompt(doc: DocType, version: str, output_mode: str) -> str:
    s = SYSTEM[(doc.name, version)]
    if output_mode == "free":
        s += ("\n\nReply with only a JSON object (no prose) with exactly these keys and types:\n"
              + json.dumps(json_schema(doc), indent=1))
    return s


def user_parts(record: dict, input_mode: str, ocr_text: str | None) -> list[dict]:
    parts: list[dict] = []
    if "image" in input_mode:
        parts += [{"type": "image", "path": p} for p in record["pages"]]
    if input_mode in ("ocr", "image+ocr", "gt_text"):
        label = "Transcription" if input_mode == "gt_text" else "OCR text (may contain errors)"
        parts.append({"type": "text", "text": f"{label}:\n\n{ocr_text}"})
    what = "receipt" if record["doc_type"] == "receipt" else "invoice"
    pages = f" ({len(record['pages'])} pages)" if len(record["pages"]) > 1 and "image" in input_mode else ""
    parts.append({"type": "text", "text": f"Extract the fields from this {what}{pages}."})
    return parts
