"""Synthetic invoices, generated from seeds and rendered with Pillow.

Real receipts show realistic failures; generated invoices let one thing change
at a time. Gold labels come from the generator, so they are exact.

A first version with clean layouts was solved: the smallest model tested got
every field of 100 dev invoices right. So each invoice now carries the traps
real invoices have, each one on a share of documents:

- number formats 1,234.56 / 1.234,56 / 1 234,56 / 1'234.56 / 1,23,456.78, and
  day-first vs month-first dates
- "$" with only the address saying which dollar
- a short brand in the header and the legal name only in the footer
- purchase order, customer and order numbers next to the invoice number;
  order, delivery and tax-point dates next to the issue date
- the due date as a date, as payment terms only, or not given
- two tax rates, or GST split into CGST and SGST
- shipping charges; a deposit already paid, with a balance due
- long tables over several pages, wrapped descriptions, units after quantities
- scan wear: rotation, blur, noise, low contrast, low resolution, JPEG

Fonts are the DejaVu files that ship with matplotlib, so the same seed renders
the same image on any machine with the same matplotlib.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import random
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
W, H = 1240, 1754  # A4 at 150 dpi

MONTHS = {
    "en": "January February March April May June July August September October November December".split(),
    "de": "Januar Februar März April Mai Juni Juli August September Oktober November Dezember".split(),
    "fr": "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split(),
}

LABELS = {
    "en": dict(title="INVOICE", number="Invoice No.", date="Invoice date", due="Due date", bill_to="Bill to",
               desc="Description", qty="Qty", unit="Unit price", amount="Amount", subtotal="Subtotal",
               discount="Discount", shipping="Shipping", total="Total", balance="Balance due", total_due="Total due",
               paid="Deposit received {d}", terms="Payment terms: Net {n} days", delivery="Delivery date",
               order="Order date", taxpoint="Tax point", po="PO number", cust="Customer no.", ref="Order ref.",
               page="Page {i} of {n}", exempt="Tax exempt", cont="continued", sku="Item"),
    "de": dict(title="RECHNUNG", number="Rechnungsnr.", date="Rechnungsdatum", due="Fällig am", bill_to="Rechnung an",
               desc="Beschreibung", qty="Menge", unit="Einzelpreis", amount="Betrag", subtotal="Zwischensumme",
               discount="Rabatt", shipping="Versandkosten", total="Gesamtbetrag", balance="Restbetrag",
               total_due="Gesamtbetrag", paid="Anzahlung erhalten {d}", terms="Zahlbar innerhalb von {n} Tagen netto",
               delivery="Lieferdatum", order="Bestelldatum", taxpoint="Leistungsdatum", po="Bestellnr.",
               cust="Kundennr.", ref="Auftragsnr.", page="Seite {i} von {n}", exempt="Steuerfrei",
               cont="Fortsetzung", sku="Art.-Nr."),
    "fr": dict(title="FACTURE", number="N° de facture", date="Date de facture", due="Échéance", bill_to="Facturé à",
               desc="Désignation", qty="Qté", unit="Prix unitaire", amount="Montant", subtotal="Sous-total HT",
               discount="Remise", shipping="Frais de port", total="Total TTC", balance="Reste à payer",
               total_due="Total TTC", paid="Acompte reçu {d}", terms="Paiement à {n} jours", delivery="Date de livraison",
               order="Date de commande", taxpoint="Date de prestation", po="N° de commande", cust="N° client",
               ref="Réf. commande", page="Page {i} sur {n}", exempt="Exonéré de TVA", cont="suite", sku="Réf."),
}


@dataclass(frozen=True)
class Locale:
    code: str
    lang: str
    currencies: tuple[tuple[str, str, int], ...]  # (ISO code, printed prefix, weight)
    taxes: tuple[tuple[str, float], ...]           # (label, rate) options for one rate
    split_tax: tuple[tuple[str, float], ...] | None  # two taxes on every line (CGST + SGST)
    second_rate: tuple[str, float] | None           # a reduced rate some lines get
    num: str
    dates: tuple[str, ...]
    address: str
    units: tuple[str, str]  # (hours, pieces)


LOCALES = [
    Locale("en_US", "en", (("USD", "$", 1),), (("Sales tax", 0.0725), ("Sales tax", 0.0825)), None, None,
           "us", ("mdy", "long_en", "mdy"), "1180 Harbor Way, Oakland, CA 94607, USA", ("hrs", "ea")),
    Locale("en_GB", "en", (("GBP", "£", 1),), (("VAT", 0.20),), None, ("VAT", 0.05),
           "us", ("dmy", "long_en_d", "dmy"), "14 Mill Lane, Leeds LS1 4DY, United Kingdom", ("hrs", "pcs")),
    Locale("en_CA", "en", (("CAD", "CAD ", 1), ("CAD", "C$", 1), ("CAD", "$", 2)), (("HST", 0.13),), None, None,
           "us", ("iso", "long_en", "dmy"), "220 King St W, Toronto, ON M5H 1K4, Canada", ("hrs", "ea")),
    Locale("en_AU", "en", (("AUD", "A$", 1), ("AUD", "AUD ", 1), ("AUD", "$", 2)), (("GST", 0.10),), None, None,
           "us", ("dmy", "long_en_d"), "Level 3, 120 Collins St, Melbourne VIC 3000, Australia", ("hrs", "ea")),
    Locale("en_IN", "en", (("INR", "Rs. ", 1), ("INR", "INR ", 1), ("INR", "₹", 1)), (("IGST", 0.18),),
           (("CGST", 0.09), ("SGST", 0.09)), None, "in", ("dmy", "long_en_d", "dmy_dash"),
           "Plot 42, MIDC Andheri East, Mumbai 400093, India", ("hrs", "nos")),
    Locale("ja_JP", "en", (("JPY", "¥", 2), ("JPY", "JPY ", 1)), (("Consumption tax", 0.10),), None, ("Consumption tax", 0.08),
           "jp", ("iso", "ymd_slash"), "2-11-3 Shibaura, Minato-ku, Tokyo 105-0023, Japan", ("h", "pcs")),
    Locale("de_DE", "de", (("EUR", "€", 2), ("EUR", "EUR ", 1)), (("MwSt.", 0.19),), None, ("MwSt.", 0.07),
           "eu", ("dmy_dot", "long_de"), "Hafenstraße 12, 20457 Hamburg, Deutschland", ("Std.", "Stk.")),
    Locale("de_CH", "de", (("CHF", "CHF ", 1), ("CHF", "Fr. ", 1)), (("MWST", 0.081),), None, ("MWST", 0.026),
           "ch", ("dmy_dot",), "Seestrasse 41, 8002 Zürich, Schweiz", ("Std.", "Stk.")),
    Locale("fr_FR", "fr", (("EUR", "€", 1), ("EUR", "EUR ", 1)), (("TVA", 0.20),), None, ("TVA", 0.10),
           "fr", ("dmy", "long_fr"), "8 rue de la Paix, 69002 Lyon, France", ("h", "pce")),
]

VENDOR_A = "Lindqvist Harbourline Ostrava Meridian Calder Brightwater Kestrel Falkner Northgate Vantor Elmsley Ardent Quarry Saffron Corvid Halden Tamsin Wexford Brandt Okafor Ferreira Marlowe Szabo Iverson".split()
VENDOR_B = "Tooling Supplies Logistics Analytics Print Works Engineering Catering Systems Labs Freight Studio Hardware Textiles Consulting Electrical".split()
SUFFIX = {"en": ["Ltd", "LLC", "Inc.", "Pty Ltd", "Pvt. Ltd.", "Co."], "de": ["GmbH", "AG", "KG"], "fr": ["SARL", "SAS", "SA"]}
CUSTOMERS = "Bluefin Hotels|Orchard Lane Dental|Pinecrest School|Nordlicht Bau|Atelier Rive Gauche|Copperleaf Cafe|Summit Physio|Riverside Clinic".split("|")
ITEMS = [
    ("Consulting, senior engineer", "h"), ("On-site support visit", "u"), ("A4 copy paper, box of 5 reams", "u"),
    ("Steel bracket M8", "u"), ("Cloud hosting, monthly", "u"), ("Design workshop", "h"),
    ("Pallet racking, 2 m bay", "u"), ("Hex bolts M10 (pack of 100)", "u"), ("LED panel 600x600", "u"),
    ("Catering, lunch per person", "u"), ("Annual software licence", "u"), ("Cable Cat6, 305 m", "u"),
    ("Data analysis", "h"), ("Safety gloves, size L", "u"), ("Printer toner, black", "u"),
    ("Translation, per 1,000 words", "u"), ("Office chair, mesh", "u"), ("Training session", "h"),
    ("Hydraulic hose 1/2in, 10 m", "u"), ("Server rack shelf", "u"), ("Project management", "h"),
    ("Label printer ribbon", "u"), ("Packaging foam inserts", "u"), ("Wall bracket, heavy duty", "u"),
    ("Network switch, 24 port", "u"), ("Ergonomic keyboard", "u"), ("Code review", "h"), ("Site survey", "h"),
    ("Coffee beans, 1 kg", "u"), ("Cleaning service", "h"), ("Printed brochures, A5 (500)", "u"),
    ("Fire extinguisher service", "u"), ("Bookkeeping", "h"), ("Ink cartridge, cyan", "u"),
]
DETAILS = ["ref. PO {po}, delivered to site B", "as agreed in quote Q-{q}", "period {m1} to {m2}",
           "incl. installation and testing", "colour: graphite", "batch {q}"]


def _q(x: Decimal, places: int) -> Decimal:
    return x.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def fmt_amount(x: Decimal, style: str) -> str:
    places = 0 if style == "jp" else 2
    x = _q(x, places)
    neg = x < 0
    x = abs(x)
    s = f"{x:.{places}f}"
    ip, _, dp = s.partition(".")
    if style == "in":
        head, tail = ip[:-3], ip[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        ip = ",".join(groups + [tail]) if groups else tail
    else:
        groups = []
        while len(ip) > 3:
            groups.insert(0, ip[-3:])
            ip = ip[:-3]
        groups.insert(0, ip)
        sep = {"us": ",", "jp": ",", "eu": ".", "fr": " ", "ch": "'"}[style]
        ip = sep.join(groups)
    dec = {"eu": ",", "fr": ","}.get(style, ".")
    out = ip + (dec + dp if places else "")
    return ("-" if neg else "") + out


def fmt_date(d: dt.date, style: str) -> str:
    if style == "mdy":
        return d.strftime("%m/%d/%Y")
    if style == "dmy":
        return d.strftime("%d/%m/%Y")
    if style == "dmy_dash":
        return d.strftime("%d-%m-%Y")
    if style == "dmy_dot":
        return d.strftime("%d.%m.%Y")
    if style == "iso":
        return d.isoformat()
    if style == "ymd_slash":
        return d.strftime("%Y/%m/%d")
    if style == "long_en":
        return f"{MONTHS['en'][d.month - 1]} {d.day}, {d.year}"
    if style == "long_en_d":
        return f"{d.day} {MONTHS['en'][d.month - 1]} {d.year}"
    if style == "long_de":
        return f"{d.day}. {MONTHS['de'][d.month - 1]} {d.year}"
    if style == "long_fr":
        return f"{d.day} {MONTHS['fr'][d.month - 1]} {d.year}"
    raise ValueError(style)


def make_invoice(seed: int) -> dict:
    """The invoice as data: what gets printed, and the gold labels."""
    r = random.Random(seed)
    loc = r.choice(LOCALES)
    code, sym, _ = r.choices(loc.currencies, weights=[c[2] for c in loc.currencies])[0]
    places = 0 if loc.num == "jp" else 2
    brand = r.choice(VENDOR_A)
    vendor = f"{brand} {r.choice(VENDOR_B)} {r.choice(SUFFIX[loc.lang])}"
    brand_only_header = r.random() < 0.3
    customer = r.choice(CUSTOMERS)
    year = r.choice([2025, 2026])
    number = r.choice([f"INV-{year}-{r.randrange(1, 9999):04d}", f"{r.randrange(10000, 99999)}",
                       f"A/{r.randrange(100, 2000)}/{year % 100}", f"{loc.code[-2:]}{r.randrange(100000, 999999)}"])
    distractor_ids = []
    if r.random() < 0.5:
        distractor_ids.append(("po", f"PO-{r.randrange(1000, 99999)}"))
    if r.random() < 0.4:
        distractor_ids.append(("cust", f"{r.randrange(10000, 99999)}"))
    if r.random() < 0.25:
        distractor_ids.append(("ref", f"SO{r.randrange(100000, 999999)}"))
    issue = dt.date(year, 1, 1) + dt.timedelta(days=r.randrange(0, 360))
    date_style = r.choice(loc.dates)
    due_mode = r.choices(["date", "terms", "none"], weights=[50, 35, 15])[0]
    net = r.choice([7, 14, 15, 30, 45, 60])
    due = issue + dt.timedelta(days=net) if due_mode in ("date", "terms") else None
    other_dates = []
    for kind, p in (("order", 0.3), ("delivery", 0.35), ("taxpoint", 0.15 if loc.lang == "en" else 0.0)):
        if r.random() < p:
            other_dates.append((kind, issue - dt.timedelta(days=r.randrange(1, 25))))

    n_items = r.choices(range(1, 31), weights=[6, 8, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1])[0]
    picked = [r.choice(ITEMS) for _ in range(n_items)]
    use_units = r.random() < 0.3
    two_rates = loc.second_rate is not None and r.random() < 0.25
    split = loc.split_tax is not None and r.random() < 0.6
    exempt = r.random() < 0.08
    tax_label, rate = r.choice(loc.taxes)
    lines = []
    for desc, kind in picked:
        if kind == "h":
            qty = Decimal(r.choice(["0.5", "1", "1.5", "2", "4", "7.5", "8", "12.25", "16", "37.5"]))
        else:
            qty = Decimal(r.choice([1, 1, 1, 2, 3, 4, 5, 6, 10, 12, 20, 25, 50, 120]))
        base = r.choice([4, 9, 18, 35, 60, 120, 240, 480, 950, 1800])
        unit = _q(Decimal(str(base * r.uniform(0.6, 1.6))), places)
        if loc.num == "jp":
            unit = _q(unit * 140, 0)
        elif loc.num == "in":
            unit = _q(unit * 80, 2)
        detail = None
        if r.random() < 0.25:
            m1 = issue.replace(day=1)
            detail = r.choice(DETAILS).format(po=r.randrange(1000, 9999), q=r.randrange(100, 999),
                                              m1=fmt_date(m1, date_style), m2=fmt_date(issue, date_style))
        line_rate = loc.second_rate[1] if two_rates and r.random() < 0.4 else rate
        lines.append({"description": desc, "detail": detail, "quantity": qty, "unit_price": unit,
                      "line_total": _q(qty * unit, places), "kind": kind, "rate": line_rate,
                      "sku": f"{r.randrange(10, 99)}-{r.randrange(100, 999)}"})
    subtotal = sum((l["line_total"] for l in lines), Decimal(0))

    discount, disc_pct = None, None
    if not two_rates and r.random() < 0.25:
        if r.random() < 0.6:
            disc_pct = r.choice([5, 10, 15])
            discount = _q(subtotal * disc_pct / 100, places)
        else:
            discount = _q(subtotal * Decimal(str(r.uniform(0.02, 0.08))), places)
    shipping = None
    if r.random() < 0.25:
        shipping = _q(Decimal(str(r.choice([6.5, 9.9, 14.0, 25.0, 48.0, 120.0]))) * (140 if loc.num == "jp" else 80 if loc.num == "in" else 1), places)
    taxed = subtotal - (discount or 0) + (shipping or 0)

    taxes: list[tuple[str, float, Decimal]] = []
    if not exempt:
        if split:
            for lab, rt in loc.split_tax:
                taxes.append((lab, rt, _q(taxed * Decimal(str(rt)), places)))
        elif two_rates:
            for rt in sorted({l["rate"] for l in lines}, reverse=True):
                base = sum((l["line_total"] for l in lines if l["rate"] == rt), Decimal(0))
                if rt == rate:
                    base += shipping or 0
                lab = tax_label if rt == rate else loc.second_rate[0]
                taxes.append((lab, rt, _q(base * Decimal(str(rt)), places)))
        else:
            taxes.append((tax_label, rate, _q(taxed * Decimal(str(rate)), places)))
    tax = sum((t[2] for t in taxes), Decimal(0)) if taxes else None
    gross = taxed + (tax or 0)
    paid = None
    if r.random() < 0.12:
        paid = _q(gross * Decimal(r.choice(["0.2", "0.3", "0.5"])), places)
    total = gross - (paid or 0)

    def f(x):
        return float(x)

    gold = {
        "vendor_name": vendor, "invoice_number": number, "issue_date": issue.isoformat(),
        "due_date": due.isoformat() if due else None, "currency": code,
        "subtotal": f(subtotal), "discount": f(discount) if discount else None,
        "shipping": f(shipping) if shipping else None,
        "tax": f(tax) if tax is not None else None,
        "amount_paid": f(paid) if paid else None, "total": f(total),
        "line_items": [{"description": l["description"], "quantity": f(l["quantity"]),
                        "unit_price": f(l["unit_price"]), "line_total": f(l["line_total"])} for l in lines],
    }
    traps = {
        "dollar_only": sym == "$" and code != "USD", "brand_header": brand_only_header,
        "distractor_ids": len(distractor_ids), "other_dates": len(other_dates), "two_rates": two_rates,
        "split_tax": split and not exempt, "shipping": shipping is not None, "paid": paid is not None,
        "units": use_units, "details": sum(l["detail"] is not None for l in lines), "n_items": n_items,
    }
    return {
        "seed": seed, "locale": loc.code, "lang": loc.lang, "num": loc.num, "sym": sym, "code": code,
        "address": loc.address, "units": loc.units, "date_style": date_style, "due_mode": due_mode, "net": net,
        "issue": issue, "due": due, "other_dates": other_dates, "vendor": vendor, "brand": brand.upper(),
        "brand_only_header": brand_only_header, "customer": customer, "number": number,
        "distractor_ids": distractor_ids, "lines": lines, "use_units": use_units, "subtotal": subtotal,
        "discount": discount, "disc_pct": disc_pct, "shipping": shipping, "taxes": taxes, "exempt": exempt,
        "gross": gross, "paid": paid, "total": total, "layout": r.randrange(3),
        "font": r.choice(["DejaVuSans", "DejaVuSerif", "DejaVuSans"]), "size": r.choice([19, 20, 22]),
        "per_page": r.choice([10, 12, 14]), "gold": gold, "traps": traps,
    }


class Canvas:
    def __init__(self, font: str, size: int):
        self.img = Image.new("RGB", (W, H), "white")
        self.d = ImageDraw.Draw(self.img)
        self.fonts = {}
        self.font, self.size = font, size
        self.words: list[tuple[int, int, str]] = []

    def f(self, bold=False, scale=1.0):
        key = (bold, scale)
        if key not in self.fonts:
            name = self.font + ("-Bold" if bold else "") + ".ttf"
            self.fonts[key] = ImageFont.truetype(str(FONT_DIR / name), int(self.size * scale))
        return self.fonts[key]

    def text(self, x, y, s, bold=False, scale=1.0, anchor="la", grey=False):
        font = self.f(bold, scale)
        self.d.text((x, y), s, fill=(110, 110, 110) if grey else (20, 20, 20), font=font, anchor=anchor)
        w = self.d.textlength(s, font=font)
        left = x - w if anchor[0] == "r" else x - w / 2 if anchor[0] == "m" else x
        self.words.append((int(y), int(left), s))

    def width(self, s, bold=False, scale=1.0):
        return self.d.textlength(s, font=self.f(bold, scale))

    def line(self, x0, y0, x1, y1, w=1):
        self.d.line((x0, y0, x1, y1), fill=(90, 90, 90), width=w)

    def ocr_text(self) -> str:
        rows: list[list[tuple]] = []
        for w in sorted(self.words):
            if rows and abs(w[0] - rows[-1][0][0]) < 8:
                rows[-1].append(w)
            else:
                rows.append([w])
        return "\n".join("   ".join(t for _, _, t in sorted(r, key=lambda w: w[1])) for r in rows)


def render(inv: dict) -> tuple[list[Image.Image], str]:
    L = LABELS[inv["lang"]]
    num, sym = inv["num"], inv["sym"]
    ds = inv["date_style"]

    def money(x):
        return f"{sym}{fmt_amount(x, num)}"

    def qty(l):
        s = f"{l['quantity'].normalize():f}"
        s = s.replace(".", ",") if num in ("eu", "fr") else s
        if inv["use_units"]:
            s += " " + (inv["units"][0] if l["kind"] == "h" else inv["units"][1])
        return s

    lines = inv["lines"]
    per = inv["per_page"]
    chunks = [lines[i:i + per] for i in range(0, len(lines), per)] or [[]]
    pages, texts = [], []
    s = inv["size"]
    lh = int(s * 1.65)
    lay = inv["layout"]
    for pi, chunk in enumerate(chunks):
        c = Canvas(inv["font"], s)
        y = 80
        if pi == 0:
            head = inv["brand"] if inv["brand_only_header"] else inv["vendor"]
            if lay == 1:
                c.text(W // 2, y, L["title"], bold=True, scale=1.8, anchor="ma")
                y += 85
                c.text(90, y, head, bold=True, scale=1.2)
            else:
                c.text(90, y, head, bold=True, scale=1.4)
                c.text(W - 90, y, L["title"], bold=True, scale=1.6, anchor="ra")
                y += 30
            y += lh
            c.text(90, y, inv["address"], scale=0.85)
            y += lh
            meta = [(L["number"], inv["number"]), (L["date"], fmt_date(inv["issue"], ds))]
            for kind, d in inv["other_dates"]:
                meta.append((L[kind], fmt_date(d, ds)))
            if inv["due_mode"] == "date":
                meta.append((L["due"], fmt_date(inv["due"], ds)))
            for kind, v in inv["distractor_ids"]:
                meta.insert(1 if kind == "po" else len(meta), (L[kind], v))
            if lay == 2:
                my = y + lh // 2
                for k, v in meta:
                    c.text(90, my, f"{k}: {v}")
                    my += lh
                y = my + lh // 2
                c.text(90, y, L["bill_to"] + ":", bold=True)
                c.text(90 + int(c.width(L["bill_to"] + ":  ", True)), y, inv["customer"])
                y += lh * 2
            else:
                bx = W - 560
                yy = y + lh // 2
                if lay == 1:
                    c.line(bx - 20, yy - 12, W - 80, yy - 12)
                for k, v in meta:
                    c.text(bx, yy, k)
                    c.text(W - 90, yy, v, anchor="ra")
                    yy += lh
                c.text(90, y + lh, L["bill_to"], bold=True)
                c.text(90, y + lh * 2, inv["customer"])
                c.text(90, y + lh * 3, "Accounts Payable", scale=0.9)
                y = max(yy, y + lh * 4) + lh
        else:
            c.text(90, y, f'{inv["brand"]}  ·  {L["number"]} {inv["number"]}  ({L["cont"]})')
            y += lh * 2
        cols = {"desc": 90 if lay != 1 else 200, "qty": 790, "unit": 1000, "amt": W - 90}
        c.line(80, y - 10, W - 80, y - 10, 2)
        if lay == 1:
            c.text(90, y, L["sku"], bold=True)
        c.text(cols["desc"], y, L["desc"], bold=True)
        c.text(cols["qty"], y, L["qty"], bold=True, anchor="ra")
        c.text(cols["unit"], y, L["unit"], bold=True, anchor="ra")
        c.text(cols["amt"], y, L["amount"], bold=True, anchor="ra")
        y += lh
        c.line(80, y - 6, W - 80, y - 6)
        for l in chunk:
            if lay == 1:
                c.text(90, y, l["sku"])
            c.text(cols["desc"], y, l["description"])
            c.text(cols["qty"], y, qty(l), anchor="ra")
            c.text(cols["unit"], y, money(l["unit_price"]), anchor="ra")
            c.text(cols["amt"], y, money(l["line_total"]), anchor="ra")
            y += lh
            if l["detail"]:
                c.text(cols["desc"] + 20, y - 4, l["detail"], scale=0.8, grey=True)
                y += int(lh * 0.8)
            if lay == 1:
                c.line(80, y - 6, W - 80, y - 6)
        if pi == len(chunks) - 1:
            y += lh // 2
            tx = 700 if lay != 2 else 560
            c.line(tx - 20, y - 8, W - 80, y - 8, 2)
            rows = [(L["subtotal"], money(inv["subtotal"]), False)]
            if inv["discount"] is not None:
                lab = L["discount"] + (f' ({inv["disc_pct"]}%)' if inv["disc_pct"] else "")
                rows.append((lab, "-" + money(inv["discount"]), False))
            if inv["shipping"] is not None:
                rows.append((L["shipping"], money(inv["shipping"]), False))
            if inv["taxes"]:
                for lab, rt, amt in inv["taxes"]:
                    pct = f"{rt * 100:g}".replace(".", "," if num in ("eu", "fr") else ".")
                    rows.append((f"{lab} {pct}%", money(amt), False))
            else:
                rows.append((L["exempt"], "", False))
            code_suffix = f' ({inv["code"]})' if lay != 0 and not (sym == "$" and inv["code"] != "USD") else ""
            if inv["paid"] is not None:
                rows.append((L["total"] + code_suffix, money(inv["gross"]), False))
                pd_ = inv["issue"] - dt.timedelta(days=3)
                rows.append((L["paid"].format(d=fmt_date(pd_, ds)), "-" + money(inv["paid"]), False))
                rows.append((L["balance"] + code_suffix, money(inv["total"]), True))
            else:
                rows.append((L["total_due"] + code_suffix, money(inv["total"]), True))
            for k, v, bold in rows:
                c.text(tx, y + (4 if bold else 0), k, bold=bold, scale=1.08 if bold else 1.0)
                if v:
                    c.text(W - 90, y + (4 if bold else 0), v, bold=bold, scale=1.08 if bold else 1.0, anchor="ra")
                y += lh
            y += lh
            if inv["due_mode"] == "terms":
                c.text(90, y, L["terms"].format(n=inv["net"]))
                y += lh
        foot = f'{inv["vendor"]} · {inv["address"]}'
        c.text(W // 2, H - 120, foot, scale=0.72, anchor="ma", grey=True)
        c.text(W // 2, H - 95, "IBAN DE44 5001 0517 5407 3249 31 · BIC INGDDEFFXXX", scale=0.72, anchor="ma", grey=True)
        if len(chunks) > 1:
            c.text(W // 2, H - 60, L["page"].format(i=pi + 1, n=len(chunks)), scale=0.8, anchor="ma")
        pages.append(c.img)
        texts.append(c.ocr_text())
    return pages, "\n\n".join(texts)


def degrade(img: Image.Image, level: int, seed: int) -> Image.Image:
    """Level 0 is the clean render; each level adds rotation, blur, noise,
    lower contrast, a lower resolution and stronger JPEG compression."""
    if level <= 0:
        return img
    r = random.Random(seed * 31 + level)
    out = img.rotate(r.uniform(-1, 1) * 0.6 * level, resample=Image.BICUBIC, expand=False, fillcolor="white")
    scale = 1 - 0.09 * level
    small = out.resize((int(W * scale), int(H * scale)), Image.BILINEAR)
    out = small.resize((W, H), Image.BILINEAR)
    out = out.filter(ImageFilter.GaussianBlur(0.3 * level))
    out = ImageEnhance.Contrast(out).enhance(1 - 0.09 * level)
    arr = np.asarray(out).astype(np.float32)
    arr += np.random.default_rng(seed + level).normal(0, 3.5 * level, arr.shape[:2])[..., None]
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=max(15, 92 - 15 * level))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


SPLITS = {"dev": range(1000, 1100), "test": range(2000, 2100)}
WEAR = [0, 1, 1, 2, 2, 3, 3, 4]  # wear level drawn per document in dev and test
DEGRADE_SEEDS = range(3000, 3040)
DEGRADE_LEVELS = range(6)


def _write(pages: list[Image.Image], text: str, img_dir: Path, stem: str, level: int) -> tuple[list[str], str]:
    # Clean renders as PNG; worn ones as JPEG, since the wear already
    # includes JPEG compression and noise makes PNGs huge.
    paths = []
    for i, p in enumerate(pages):
        if level == 0:
            path = img_dir / f"{stem}_p{i + 1}.png"
            p.save(path)
        else:
            path = img_dir / f"{stem}_p{i + 1}.jpg"
            p.save(path, quality=95)
        paths.append(path.as_posix())
    tpath = img_dir / f"{stem}.gt.txt"
    tpath.write_text(text, encoding="utf-8")
    return paths, tpath.as_posix()


def _meta(inv: dict, n_pages: int, level: int) -> dict:
    return {k: inv[k] for k in ("seed", "locale", "num", "date_style", "due_mode", "layout")} | {
        "pages": n_pages, "degrade": level, "traps": inv["traps"]}


def build(out_dir: Path, gold_dir: Path) -> None:
    gold_dir.mkdir(parents=True, exist_ok=True)
    for split, seeds in SPLITS.items():
        img_dir = out_dir / split
        img_dir.mkdir(parents=True, exist_ok=True)
        recs = []
        for seed in seeds:
            inv = make_invoice(seed)
            pages, text = render(inv)
            level = random.Random(seed).choice(WEAR)
            pages = [degrade(p, level, seed + i) for i, p in enumerate(pages)]
            paths, tpath = _write(pages, text, img_dir, f"{seed}", level)
            recs.append({"id": f"inv-{split}-{seed}", "set": "invoices", "split": split, "doc_type": "invoice",
                         "pages": paths, "gt_text": tpath, "gold": inv["gold"], "excluded": {},
                         "meta": _meta(inv, len(paths), level)})
        with open(gold_dir / f"invoices_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"invoices {split}: {len(recs)}")
    img_dir = out_dir / "degrade"
    img_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for seed in DEGRADE_SEEDS:
        inv = make_invoice(seed)
        clean, text = render(inv)
        for level in DEGRADE_LEVELS:
            pages = [degrade(p, level, seed + i) for i, p in enumerate(clean)]
            paths, tpath = _write(pages, text, img_dir, f"{seed}_l{level}", level)
            recs.append({"id": f"inv-degrade-{seed}-l{level}", "set": "degrade", "split": "test", "doc_type": "invoice",
                         "pages": paths, "gt_text": tpath, "gold": inv["gold"], "excluded": {},
                         "meta": _meta(inv, len(paths), level)})
    with open(gold_dir / "degrade_test.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"degradation set: {len(recs)}")
