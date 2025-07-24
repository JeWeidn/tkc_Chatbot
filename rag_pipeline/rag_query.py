# -*- coding: utf-8 -*-
"""RAG-Pipeline – Query Script
Hybrid-Retrieval (BM25 + Dense + Self-Query) inkl. LLM-Kompression + Begründung.
Aufruf:
    python rag_query.py "Deine Frage …"
"""
from __future__ import annotations

import os, sys, json
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.query_constructor.base import AttributeInfo

sys.stdout.reconfigure(encoding="utf-8")  # Konsolen-UTF-8

# ──────────────────────────────
# 1)  Key & Pfade
# ──────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
BACKEND    = BASE_DIR.parent / "backend"
VECTOR_DIR = BACKEND / "vector_db"
PDF_PATH   = BACKEND / "docs" / "mhb_wiing_BSc_de_aktuell.pdf"
JSON_PATH  = PDF_PATH.with_name("TestText.json")

load_dotenv(BACKEND / ".env")
api_key = os.getenv("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY fehlt")

# ──────────────────────────────
# 2)  Frage
# ──────────────────────────────
question = " ".join(sys.argv[1:]).strip() or sys.exit("Frage fehlt!")

# ──────────────────────────────
# 3)  Vector-DB + Dense-Retriever
# ──────────────────────────────
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectordb   = Chroma(persist_directory=str(VECTOR_DIR), embedding_function=embeddings)

dense_retriever = vectordb.as_retriever(search_kwargs={"k": 12})

# ──────────────────────────────
# 3b)  BM25-Retriever (PDF + JSON)
# ──────────────────────────────
loader = PyPDFLoader(str(PDF_PATH))
bm25_docs = loader.load()   # PDF-Seiten

if JSON_PATH.exists():
    with open(JSON_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    from langchain.docstore.document import Document
    bm25_docs.extend(Document(page_content=e["text"], metadata=e) for e in meta)

bm25 = BM25Retriever.from_documents(bm25_docs)
bm25.k = 20  # großzügiger Recall

# ──────────────────────────────
# 3c)  Self-Query-Retriever (Metadaten-Filter)
# ──────────────────────────────
metadata_field_info = [
    AttributeInfo(name="ects_lp",      type="float",  description="Leistungspunkte"),
    AttributeInfo(name="responsibility", type="string", description="Verantwortlicher Dozent"),
]

self_query = SelfQueryRetriever.from_llm(
    ChatOpenAI(model="gpt-4", openai_api_key=api_key, temperature=0),
    vectordb,
    "Modul- und Teilleistungsbeschreibungen",
    metadata_field_info,
    verbose=False,
)

# ──────────────────────────────
# 3d)  Ensemble + Kompressor
# ──────────────────────────────
hybrid = EnsembleRetriever(
    retrievers=[bm25, dense_retriever, self_query],
    weights=[0.25, 0.45, 0.30],
)

compressor = LLMChainExtractor.from_llm(ChatOpenAI(model="gpt-4", openai_api_key=api_key, temperature=0))
retriever  = ContextualCompressionRetriever(base_retriever=hybrid, base_compressor=compressor)

# ──────────────────────────────
# 4)  Prompt & QA-Chain
# ──────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein Experte und Studienberater deines Studiengangs Wirtschaftsingenieurwesen. Erkenne bei der Frage, ob es sich um eine allgemeine Frage zum Studiengang handelt oder um eine spezifische Frage zu einem Modul oder Teilleistung und nutze dabei entsprechend die dazugegebenen Informationen für deine Antwort."),
    ("user",   "Kontext:\n{context}\n\nFrage: {question}"),
])

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4", openai_api_key=api_key),
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

# ─── 4b)  QA-Aufruf ──────────────────────────────────────────────
result = qa_chain.invoke({"query": question})

# ─── 4c)  Begründung generieren ─────────────────────────────────
raw_docs = result.pop("source_documents", [])

# Quellen für Frontend (ohne justification)
result["source_documents"] = [
    {
        "page":   d.metadata.get("page", "–"),
        "source": d.metadata.get("source", d.metadata.get("title", "meta")),
    }
    for d in raw_docs
]

# Max. 3000 Zeichen Kontext für Erklär-LLM
context_excerpt = "\n\n".join(d.page_content for d in raw_docs)[:3000]

explainer_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0)
explainer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Du bist ein Tutor. Fasse in höchstens 3 Sätzen zusammen, warum die gegebene Antwort aufgrund des Kontexts plausibel ist. Nenne keine neuen Informationen.",
    ),
    (
        "user",
        "Frage: {question}\nAntwort: {answer}\nKontext:\n{context}",
    ),
])

justification_msg = explainer_llm.invoke(
    explainer_prompt.format(question=question, answer=result["result"], context=context_excerpt)
)

result["justification"] = justification_msg.content.strip()

# ──────────────────────────────
# 5)  Ergebnis als JSON für Node
# ──────────────────────────────
print(json.dumps(result, ensure_ascii=False))
