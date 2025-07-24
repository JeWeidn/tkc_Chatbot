"""
Metadaten‑Extractor für Modulhandbuch‑PDF (v5.2)
=============================================

**Fixes & Optimierungen**
1.  **Context‑Limit‑Guard** – das an GPT‑4o übergebene `block` wird vor dem
    Call auf **max. 12 000 Zeichen** gekürzt; kein 400‑Fehler wegen 128 k Token
    mehr.
2.  **LP‑Regex** ab v5.1 (greift auch < 5 LP) bleibt erhalten.
3.  Vollständiger Code ohne Ellipsen – sofort copy‑&‑run.

Aufruf (Beispiel):
```bash
python extract_metadata.py                    # modules.json + teile.json
```
```
pip install openai langchain langchain-community pydantic unstructured python-dotenv tqdm
```
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
from dotenv import load_dotenv

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

# ---------------------------------------------------------------------------
# Paths & Models
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = BASE_DIR / "docs" / "mhb_wiing_BSc_de_aktuell.pdf"
DEFAULT_MOD_OUT = BASE_DIR / "modules.json"
DEFAULT_TL_OUT = BASE_DIR / "teile.json"

class BaseMeta(BaseModel):
    module_id: str
    title: str
    page: int
    ects_lp: float | None = None
    language: str | None = None

class ModuleMeta(BaseMeta):
    responsibility: str | None = None
    institution: str | None = None
    part_of: str | None = None
    turnus: str | None = None
    mandatory_components: str | None = None
    learning_objectives: str | None = None
    content: str | None = None
    notes: str | None = None

class PartMeta(BaseMeta):
    component_type: str | None = None
    lectures: str | None = None
    exams: str | None = None
    assessment: str | None = None
    prerequisites: str | None = None
    organizational: str | None = None
    learning_goals: str | None = None

# ---------------------------------------------------------------------------
# Heading Map & Normalisation
# ---------------------------------------------------------------------------
RAW_SECTION_MAP_MOD = {
    "Bestandteil von": "part_of",
    "Turnus": "turnus",
    "Pflichtbestandteile": "mandatory_components",
    "Qualifikationsziele": "learning_objectives",
    "Inhalt": "content",
    "Anmerkungen": "notes",
}
RAW_SECTION_MAP_TL = {
    "Teilleistungsart": "component_type",
    "Lehrveranstaltungen": "lectures",
    "Prüfungsveranstaltungen": "exams",
    "Erfolgskontrolle": "assessment",
    "Voraussetzungen": "prerequisites",
    "Modellierte Voraussetzungen": "prerequisites",
    "Organisatorisches": "organizational",
    "Lernziele": "learning_goals",
}
RAW_SECTION_MAP = {**RAW_SECTION_MAP_MOD, **RAW_SECTION_MAP_TL}

def _norm(h: str) -> str:
    h = h.lower()
    h = re.sub(r"\(.*?\)", "", h)
    h = re.sub(r"[^a-zäöüß]", "", h)
    return h.strip()

SECTION_MAP = {_norm(k): v for k, v in RAW_SECTION_MAP.items()}
SECTION_PATTERN = re.compile(rf"^({'|'.join(map(re.escape, SECTION_MAP.keys()))})[:\s]?", re.I)

# ---------------------------------------------------------------------------
# Page Load & Block Split
# ---------------------------------------------------------------------------
MODULE_PAGES = range(25, 134)  # inclusive 133
TL_PAGES = range(134, 490)     # inclusive 489

MOD_TITLE = re.compile(r"6\.\d+\s+.*?\[M-[A-Z]{3,4}-\d{5}\]", re.I)
TL_TITLE  = re.compile(r"7\.\d+\s+.*?\[T-[A-Z]{3,4}-\d{5}\]", re.I)
CODE_ANY = re.compile(r"[MT]-[A-Z]{3,4}-\d{5}")

def load_pages(pdf: Path) -> List[str]:
    return [p.page_content for p in PyPDFLoader(str(pdf)).load()]


def find_blocks(pages: List[str]) -> Tuple[List[Tuple[str,int]], List[Tuple[str,int]]]:
    modules, parts = [], []
    buf: List[str] = []
    buf_page: int | None = None
    target: str | None = None

    def flush():
        if not buf:
            return
        block = "\n".join(buf)
        if target == "M":
            modules.append((block, buf_page))
        elif target == "T":
            parts.append((block, buf_page))
        buf.clear()

    for idx, page_text in enumerate(pages, start=1):
        if idx not in MODULE_PAGES and idx not in TL_PAGES:
            continue
        for line in page_text.splitlines():
            if MOD_TITLE.match(line):
                flush(); target="M"; buf_page=idx
            elif TL_TITLE.match(line):
                flush(); target="T"; buf_page=idx
            elif target is None and CODE_ANY.search(line):
                target = "T" if "T-" in line else "M"; buf_page = idx
            buf.append(line)
    flush()
    return modules, parts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_lp(raw: str | float | int | None) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int,float)):
        return float(raw)
    m = re.search(r"\d+(?:[.,]\d+)?", raw)
    return float(m.group().replace(",",".")) if m else None

LP_ANY = re.compile(r"(\d{1,2}(?:[.,]\d+)?)\s*(?:LP|ECTS)", re.I)


def regex_lp(block: str) -> Dict[str, Any]:
    m = re.search(r"Leistungspunkte\s*(\d{1,2}(?:[.,]\d)?)", block, re.I)
    if m:
        return {"ects_lp": clean_lp(m.group(1))}
    m2 = LP_ANY.search(block)
    return {"ects_lp": clean_lp(m2.group(1))} if m2 else {}


def head_fields(block: str) -> Dict[str,str]:
    out: Dict[str,str] = {}
    for line in block.splitlines()[:12]:
        low = line.lower()
        if low.startswith("verantwortung"):
            out["responsibility"] = line.split(":",1)[-1].strip()
        elif low.startswith("einrichtung"):
            out["institution"] = line.split(":",1)[-1].strip()
    return out


def parse_sections(block: str) -> Dict[str,str]:
    data: Dict[str, List[str]] = {}
    current=None
    for raw in block.splitlines():
        line=raw.lstrip("•-*–● \t")
        m=SECTION_PATTERN.match(line)
        if m:
            current = SECTION_MAP.get(_norm(m.group(1)))
            after=line[m.end():].strip()
            if current and after:
                data.setdefault(current,[]).append(after)
        elif current:
            data.setdefault(current,[]).append(line.strip())
    return {k:" \n ".join(v).strip() for k,v in data.items() if v}

# ---------------------------------------------------------------------------
# LLM Back‑fill with context guard
# ---------------------------------------------------------------------------

def build_schema(missing: List[str]) -> Dict[str, Any]:
    props={f:{"type":"string"} for f in missing}; props.update({"module_id":{"type":"string"},"title":{"type":"string"}})
    return {"name":"extract","description":"fill","parameters":{"type":"object","properties":props,"required":["module_id","title"]}}


def llm_fill(block: str, partial: Dict[str, Any], client: OpenAI) -> Dict[str, Any]:
    """Füllt fehlende Felder via GPT‑4o.  
    * Kürzt *block* auf 12 000 Zeichen, um das Kontext‑Limit sicher einzuhalten.  
    * Wenn das Modell **keinen** Function‑Call liefert (tool_calls == None),
      versucht es, direkt den JSON‑String aus `message.content` zu lesen.
    * Schlägt auch das fehl oder das JSON ist ungültig, wird *partial*
      unverändert zurückgegeben (kein Hard‑Error mehr).
    """
    missing = [k for k in (set(ModuleMeta.model_fields) | set(PartMeta.model_fields)) if k not in partial or partial[k] in (None, "")]
    if not missing:
        return partial

    schema = build_schema(missing)
    short_block = block[:12000]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return JSON for missing fields"},
                {"role": "user", "content": short_block},
            ],
            tools=[{"type": "function", "function": schema}],
            tool_choice="auto",
            response_format={"type": "json_object"},
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            raw_json = msg.tool_calls[0].function.arguments
        else:
            raw_json = msg.content or "{}"
        enriched = json.loads(raw_json)
    except Exception as e:  # Netzwerkfehler, JSON‑Decode etc.
        print(f"[33m⚠ LLM‑fill skipped ({e})[0m")
        return partial  # graceful degradation

    enriched.update(partial)
    enriched["ects_lp"] = clean_lp(enriched.get("ects_lp"))
    return enriched

# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def process(pdf: Path, mod_out: Path, tl_out: Path, show_bar: bool):
    load_dotenv(); client=OpenAI()
    pages=load_pages(pdf)
    mod_blocks, tl_blocks=find_blocks(pages)

    result_mod: List[Dict[str,Any]]=[]; result_tl: List[Dict[str,Any]]=[]
    iterator = tqdm(mod_blocks+tl_blocks, desc="Extract", unit="block") if show_bar else (mod_blocks+tl_blocks)

    for block,page in iterator:
        rec={"page":page}; rec.update(regex_lp(block)); rec.update(parse_sections(block)); rec.update(head_fields(block))
        rec=llm_fill(block,rec,client)
        try:
            if rec["module_id"].startswith("M-"):
                result_mod.append(ModuleMeta(**rec).model_dump())
            else:
                result_tl.append(PartMeta(**rec).model_dump())
        except Exception as e:
            print(f"\033[33m⚠ Skip {rec.get('module_id','?')} ({e})\033[0m")

    mod_out.parent.mkdir(parents=True,exist_ok=True)
    tl_out.parent.mkdir(parents=True,exist_ok=True)
    with mod_out.open("w",encoding="utf-8") as f: json.dump(result_mod,f,ensure_ascii=False,indent=2)
    with tl_out.open("w",encoding="utf-8") as f: json.dump(result_tl,f,ensure_ascii=False,indent=2)

    print(f"\n✅ {len(result_mod)} Module → {mod_out}\n✅ {len(result_tl)} Teilleistungen → {tl_out}\n")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__=="__main__":
    ap=argparse.ArgumentParser(description="Extractor v5.2")
    ap.add_argument("--pdf",type=Path,default=DEFAULT_PDF)
    ap.add_argument("--modules-out",type=Path,default=DEFAULT_MOD_OUT)
    ap.add_argument("--parts-out",type=Path,default=DEFAULT_TL_OUT)
    ap.add_argument("--no-bar",action="store_true")
    args=ap.parse_args()
    if not args.pdf.exists():
        print(f"❌ PDF not found: {args.pdf}",file=sys.stderr); sys.exit(1)
    process(args.pdf,args.modules_out,args.parts_out,show_bar=not args.no_bar)
