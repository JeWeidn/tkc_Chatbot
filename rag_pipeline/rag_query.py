import os, sys, json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
import os, sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # ← NEU

# ─── 1)  Key laden ───────────────────────────────────────────────
load_dotenv(Path(__file__).resolve().parent.parent / "backend" / ".env")
api_key = os.getenv("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY fehlt")

# ─── 2)  Frage einlesen ───────────────────────────────────────────
question = " ".join(sys.argv[1:]).strip() or sys.exit("Frage fehlt!")

# ─── 3)  Vektor-DB + Embeddings ─────────────────────────────────────
VECTOR_DIR = Path(__file__).resolve().parent.parent / "backend" / "vector_db"
vectordb = Chroma(
    persist_directory=str(VECTOR_DIR),
    embedding_function=OpenAIEmbeddings(openai_api_key=api_key),
)

dense_retriever = vectordb.as_retriever(search_kwargs={"k": 12})

# ─── 3b)  BM25-Retriever aus PDF (einmal rendern, ~0.5 s) ──────────
PDF_PATH = (
    Path(__file__).resolve().parent.parent
    / "backend"
    / "docs"
    / "mhb_wiing_BSc_de_aktuell.pdf"
)
bm25_docs = PyPDFLoader(str(PDF_PATH)).load()

bm25 = BM25Retriever.from_documents(bm25_docs)
bm25.k = 20  # BM25 darf ruhig viele Rohtreffer liefern

# ─── 3c)  Ensemble-Hybrid ──────────────────────────────────────────
hybrid = EnsembleRetriever(               # 〈– Variable heißt jetzt hybrid
    retrievers=[bm25, dense_retriever],
    weights=[0.35, 0.65]
)

# ------------------------------------------------------------------
#  3d) LLM-Kompressor (LLMChainExtractor)
# ------------------------------------------------------------------
compressor = LLMChainExtractor.from_llm(
    ChatOpenAI(model="gpt-4", openai_api_key=api_key, temperature=0)
)

retriever = ContextualCompressionRetriever(
    base_retriever=hybrid,
    base_compressor=compressor,
)

# ─── 4)  Prompt & QA-Chain ─────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein Tutor. Antworte kurz und auf Deutsch."),
    ("user",   "Kontext:\n{context}\n\nFrage: {question}")
])

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4", openai_api_key=api_key),
    retriever=retriever,
    # Prompt bleibt in chain_type_kwargs
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)


result = qa_chain.invoke({"query": question})

# ─── 5)  Quellen serialisierbar machen ─────────────────────────────
def doc_to_dict(doc):
    return {
        "page":   doc.metadata.get("page", "–"),
        "source": Path(doc.metadata.get("source", "")).name
    }

result["source_documents"] = [doc_to_dict(d) for d in result.pop("source_documents", [])]
print(json.dumps(result, ensure_ascii=False))
