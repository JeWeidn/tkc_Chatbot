# rag_pipeline/create_index.py
from pathlib import Path
import os, re
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

# ──────────────────────────────
# 1) Pfade & Key
# ──────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent          # …/rag_pipeline
PDF_PATH   = BASE_DIR.parent / "backend" / "docs" / "mhb_wiing_BSc_de_aktuell.pdf"
VECTOR_DIR = BASE_DIR.parent / "backend" / "vector_db"

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF nicht gefunden: {PDF_PATH}")

load_dotenv(BASE_DIR.parent / "backend" / ".env")
api_key = os.getenv("OPENAI_API_KEY") or \
          (_ for _ in ()).throw(EnvironmentError("OPENAI_API_KEY fehlt in .env"))

embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# ──────────────────────────────
# 2) PDF laden
# ──────────────────────────────
loader = PyPDFLoader(str(PDF_PATH))
raw_pages = loader.load()        # Liste von Document-Objekten (page_content + metadata)

# ──────────────────────────────
# 3) Modul-Boundary-Split
# ──────────────────────────────
module_chunks: list[str] = []
current = ""
current_page = 1

pattern = re.compile(r"^\s*(M|T)-WIWI-\d{5}")

for page_doc in raw_pages:
    for line in page_doc.page_content.splitlines():
        if pattern.match(line):
            if current:
                module_chunks.append((current, current_page))
                current = ""
                current_page = page_doc.metadata.get("page", 0) + 1
        current += line + "\n"

# letztes Modul speichern
if current:
    module_chunks.append((current, current_page))

# ──────────────────────────────
# 4) Feinsplit + Metadaten
# ──────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=80)
docs: list[Document] = []

for text, page in module_chunks:
    for d in splitter.split_text(text):
        docs.append(
            Document(
                page_content=d,
                metadata={
                    "page": page,
                    "source": PDF_PATH.name,
                    "title": d.split("\n", 1)[0].strip()[:120]  # Kopfzeile als Titel
                }
            )
        )

# ──────────────────────────────
# 5) Vektor-DB neu anlegen
# ──────────────────────────────
if VECTOR_DIR.exists():
    # alter Index löschen, um Konflikte zu vermeiden
    import shutil
    shutil.rmtree(VECTOR_DIR)

vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory=str(VECTOR_DIR)
)

print(f"✅ {len(docs)} Chunks indiziert → {VECTOR_DIR}")
