"""Tesseract OCR, cached per page next to the image."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]


def _binary() -> str:
    exe = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if exe:
        return exe
    for c in _CANDIDATES:
        if Path(c).exists():
            return c
    raise RuntimeError("Tesseract not found: install it, or set TESSERACT_CMD")


TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/4.1.0/{lang}.traineddata"


def fetch_models() -> None:
    import urllib.request
    d = Path(__file__).resolve().parents[1] / "data" / "tessdata"
    d.mkdir(parents=True, exist_ok=True)
    for lang in ("eng", "deu", "fra"):
        if not (d / f"{lang}.traineddata").exists():
            urllib.request.urlretrieve(TESSDATA_URL.format(lang=lang), d / f"{lang}.traineddata")


def ocr_record(record: dict) -> str:
    import pytesseract
    from PIL import Image

    pytesseract.pytesseract.tesseract_cmd = _binary()
    texts = []
    for p in record["pages"]:
        cache = Path(p).with_suffix(".tess.txt")
        if cache.exists():
            texts.append(cache.read_text(encoding="utf-8"))
            continue
        lang = "eng+deu+fra" if record["doc_type"] == "invoice" else "eng"
        # tessdata_fast 4.1.0 models in data/tessdata (python -m xeval data fetches them)
        tessdata = Path(__file__).resolve().parents[1] / "data" / "tessdata"
        os.environ["TESSDATA_PREFIX"] = str(tessdata)
        t = pytesseract.image_to_string(Image.open(p), lang=lang, config="--psm 4 -c preserve_interword_spaces=1")
        cache.write_text(t, encoding="utf-8")
        texts.append(t)
    return "\n\n".join(texts)
