# -*- coding: utf-8 -*-
"""RAG‑Pipeline – Index‑Builder
Erzeugt einen Hybrid‑Index (Dense + BM25) aus
1. den reinen PDF‑Chunks und
2. den strukturierten Metadaten aus *TestText.json*.

Bei jedem Lauf wird das Zielverzeichnis komplett neu angelegt.
"""
from __future__ import annotations

import os, re, json, shutil, itertools
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

# ──────────────────────────────
# 1)  Pfade & OpenAI‑Key
# ──────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent          # …/rag_pipeline
PDF_PATH   = BASE_DIR.parent / "backend" / "docs" / "mhb_wiing_BSc_de_aktuell.pdf"
JSON_PATH  = BASE_DIR.parent / "backend" / "docs" / "TestText.json"
VECTOR_DIR = BASE_DIR.parent / "backend" / "vector_db"

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF nicht gefunden: {PDF_PATH}")

load_dotenv(BASE_DIR.parent / "backend" / ".env")
api_key = os.getenv("OPENAI_API_KEY") or (
    _ for _ in ()
).throw(EnvironmentError("OPENAI_API_KEY fehlt in .env"))

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",       # neues Modell
    dimensions=1024,                      # optional: Vektor kürzen → 1024
    openai_api_key=api_key,
)

# ──────────────────────────────
# 2)  PDF laden
# ──────────────────────────────
loader = PyPDFLoader(str(PDF_PATH))
raw_pages = loader.load()   # List[Document] – pro Seite ein Doc

# ──────────────────────────────
# 3)  Modul‑Boundary‑Split (grobe Cuts)
# ──────────────────────────────
module_chunks: list[tuple[str, int]] = []
current_txt: str = ""
current_page: int = 1

pattern = re.compile(r"^\s*(M|T)-WIWI-\d{5}")

for page_doc in raw_pages:
    for line in page_doc.page_content.splitlines():
        if pattern.match(line):
            if current_txt:
                module_chunks.append((current_txt, current_page))
                current_txt = ""
                current_page = page_doc.metadata.get("page", 0) + 1
        current_txt += line + "\n"

if current_txt:
    module_chunks.append((current_txt, current_page))

# ──────────────────────────────
# 4)  Feinsplit + Metadaten
# ──────────────────────────────

# 4a)  JSON‑Metadaten → meta_docs
meta_docs: list[Document] = []
if JSON_PATH.exists():
    with open(JSON_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    for entry in meta:
        body = f"{entry['title']}\n\n{entry['text']}"
        meta_docs.append(
            Document(
                page_content=body,
                metadata={k: v for k, v in entry.items() if k != "text"} | {"doc_type": "meta"},
            )
        )

# 4b)  PDF‑Chunks → pdf_docs
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=80)
pdf_docs: list[Document] = []

for text, page in module_chunks:
    for chunk in splitter.split_text(text):
        pdf_docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "page": page,
                    "source": PDF_PATH.name,
                    "title": chunk.split("\n", 1)[0].strip()[:120],
                    "doc_type": "pdf",
                },
            )
        )

# 4c)  Quellen zusammenführen
all_docs: list[Document] = meta_docs + pdf_docs

# ──────────────────────────────
# 5)  Vector‑DB (Chroma) neu anlegen
# ──────────────────────────────
if VECTOR_DIR.exists():
    shutil.rmtree(VECTOR_DIR)

vectorstore = Chroma.from_documents(
    all_docs,
    embeddings,
    persist_directory=str(VECTOR_DIR),
)

print(f"✅ {len(all_docs)} Chunks indiziert → {VECTOR_DIR}")
