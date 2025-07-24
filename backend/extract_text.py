"""
Quick PDFTextDumper v3.7 (komplett)
====================================

*Liest eine PDF, überspringt die ersten **N Seiten** (Default 25) und speichert
jede Seite als JSON‑Objekt in **TestText.json**.*  Objekte mit identischem
`title` werden zusammengeführt; der Fließtext bleibt in `text`.

**Neu in v3.7**
* **`new_knowledge`** (leeres Array) wird **jedem** Datensatz hinzugefügt –
  vorgesehen, um später vom Chatbot erzeugtes Zusatz‑Wissen abzulegen.
* Alle bisherigen Felder bleiben erhalten.

```bash
python extract_text.py                # Standard (skip=25)
```

Abhängigkeiten: `pip install langchain-community unstructured tqdm`
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    def tqdm(it, **kw):  # type: ignore
        return it

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = BASE_DIR / "docs" / "mhb_wiing_BSc_de_aktuell.pdf"
DEFAULT_OUT = BASE_DIR / "TestText.json"
DEFAULT_SKIP = 25

# ---------------------------------------------------------------------------
# Regex‑Vorlagen
# ---------------------------------------------------------------------------
_TITLE_PREFIX = re.compile(r"^[67]\s+(?:module|modules|teilleistungen)\s*:?.*?\b", re.I)
_RESP_RGX     = re.compile(r"Verantwortung:\s*(.*?)\s*(?:\n|$)")
_INST_RGX     = re.compile(r"Einrichtung:\s*(.*?)\s*(?:\n|$)")
_PART_MOD     = re.compile(r"Bestandteil von:\s*(.*?)\s*(?:\n|$)")
_PART_TL      = re.compile(r"Bestandteil von:\s*(.*?)\s*(?=\n.*?Teilleistungsart)", re.S)
_ECTS_RGX     = re.compile(r"Leistungspunkte\s*\n\s*(\d+[.,]?\d*)")
_PWA_BLOCK    = re.compile(r"T-[A-Z]{4}-\d{5,6}.*?(?=\n\s*Erfolgskontrolle\(n\))", re.S)
_STOP_LINE    = r"\n\s*[^.,\n\s]+(?:\s+[^.,\n\s]+)?\s*\n|$"  # max 2 Wörter, kein Punkt/Komma
_ERFOLG_RGX   = re.compile(rf"Erfolgskontrolle\(n\)\s*\n(.*?)(?={_STOP_LINE})", re.S)
_QUALI_RGX    = re.compile(rf"Qualifikationsziele\s*\n(.*?)(?={_STOP_LINE})", re.S)
_VOR_RGX      = re.compile(rf"Voraussetzungen\s*\n(.*?)(?={_STOP_LINE})", re.S)
_INHALT_RGX   = re.compile(rf"(?:Inhalt|Lehrinhalt):?\s*\n(.*?)(?={_STOP_LINE})", re.S)
_ANM_RGX      = re.compile(rf"(?:Anmerkungen|Hinweise):?\s*\n(.*?)(?={_STOP_LINE})", re.S)
_SPLIT_DELIM  = re.compile(r"[\n,]+")


# ---------------------------------------------------------------------------
# Hilfs‑Funktionen
# ---------------------------------------------------------------------------

def pdf_to_pages(pdf: Path) -> list[str]:
    """Lädt eine PDF und gibt pro Seite den reinen Text zurück."""
    return [p.page_content for p in PyPDFLoader(str(pdf)).load()]


def clean_title(raw: str) -> str:
    """Bereinigt führende Kapitel‑/Zählerpräfixe in der Titelzeile."""
    return _TITLE_PREFIX.sub("", raw).strip()


def drop_first_three(text: str) -> str:
    """Entfernt die ersten drei Zeilen (Seitenkopf)."""
    parts = text.split("\n", 3)
    return parts[3].strip() if len(parts) > 3 else text.strip()


def extract_one(rx: re.Pattern[str], text: str) -> str | None:
    if (m := rx.search(text)):
        return m.group(1).strip()
    return None


def extract_multi(rx: re.Pattern[str], text: str) -> str | None:
    ms = [m.group(1).strip() for m in rx.finditer(text) if m.group(1).strip()]
    if not ms:
        return None
    return "\n\n".join(ms)


def extract_part_of(text: str, is_module: bool) -> list[str] | None:
    m = (_PART_MOD if is_module else _PART_TL).search(text)
    if not m:
        return None
    seg = m.group(1).strip()
    parts = [s.strip(" •-–\t") for s in _SPLIT_DELIM.split(seg) if s.strip()]
    return parts or None


def extract_ects(text: str) -> float | None:
    if (m := _ECTS_RGX.search(text)):
        try:
            return float(m.group(1).replace(',', '.'))
        except ValueError:
            return None
    return None


def extract_pwa(text: str) -> list[str] | None:
    if (m := _PWA_BLOCK.search(text)):
        lines = [ln.strip() for ln in m.group(0).split("\n")]
        res: list[str] = []
        expect_t = True
        for ln in lines:
            if ln.startswith("T-"):
                res.append(ln)
                expect_t = False
            elif not expect_t:  # Leerzeile / Abschnittswechsel → Ergänzungsangebot‑Marker
                res.append("Ergänzungsangebot:")
                expect_t = True
        return res or None
    return None


# ---------------------------------------------------------------------------
# Merge‑Logik
# ---------------------------------------------------------------------------

def merge(items: list[dict]) -> list[dict]:
    """Fasst Seiten mit gleichem Titel zusammen und extrahiert Metadaten."""
    buckets: dict[str, dict] = {}
    for it in items:
        b = buckets.setdefault(it["title"], {"title": it["title"], "pages": [], "texts": []})
        b["pages"].append(it["page"])
        b["texts"].append(it["text"])

    merged: list[dict] = []
    for data in buckets.values():
        pages = sorted(data["pages"])
        page_str = str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"
        combined = "\n\n".join(data["texts"]).strip()

        obj: dict[str, object] = {
            "title": data["title"],
            "page": page_str,
            "text": combined,
            "New_Knowledge": []  # ← NEU ab v3.7
        }

        if (resp := extract_one(_RESP_RGX, combined)):
            obj["responsibility"] = resp
        if (inst := extract_one(_INST_RGX, combined)):
            obj["institution"] = inst

        is_module = data["title"].lower().startswith("modul")
        if (parts := extract_part_of(combined, is_module)):
            obj["part_of"] = parts
        if (ects := extract_ects(combined)) is not None:
            obj["ects_lp"] = ects
        if is_module and (pwa := extract_pwa(combined)):
            obj["pflicht_wahl_angebot"] = pwa
        if is_module and (quali := extract_one(_QUALI_RGX, combined)):
            obj["qualifikationsziele"] = quali
        if (vor := extract_one(_VOR_RGX, combined)):
            obj["voraussetzungen"] = vor
        if (erf := extract_one(_ERFOLG_RGX, combined)):
            obj["erfolgskontrolle"] = erf
        if (inh := extract_multi(_INHALT_RGX, combined)):
            obj["inhalt"] = inh
        if (anm := extract_multi(_ANM_RGX, combined)):
            obj["anmerkungen"] = anm

        merged.append(obj)
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(pdf: Path, out: Path, skip: int):
    if not pdf.exists():
        raise FileNotFoundError(pdf)

    pages = pdf_to_pages(pdf)[skip:]
    rows: list[dict] = []
    for p_no, txt in tqdm(
        enumerate(pages, start=skip + 1),
        total=len(pages),
        desc="Extract", unit="page", dynamic_ncols=True, mininterval=0.1, ascii=True):
        txt = txt.strip()
        if not txt:
            continue
        title_line = txt.split("\n", 1)[0].strip()
        rows.append({"page": p_no, "title": clean_title(title_line), "text": drop_first_three(txt)})

    data = merge(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(data)} Einträge aus {len(rows)} Seiten → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF → merged JSON (v3.7)")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Pfad zur PDF")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Ausgabe‑JSON")
    parser.add_argument("--skip", type=int, default=DEFAULT_SKIP, help="Erste N Seiten überspringen")
    args = parser.parse_args()

    main(args.pdf, args.out, args.skip)
