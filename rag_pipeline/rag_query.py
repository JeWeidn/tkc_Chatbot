# -*- coding: utf-8 -*-
"""
RAG-Pipeline – Query Script
Hybrid-Retrieval + Cross-Encoder-Rerank + Conversational-LLM
Gibt JSON mit answer / generated_question / source_documents / justification / extracted_knowledge zurück.
"""
from __future__ import annotations

import os, sys, json, warnings, re
from pathlib import Path

import torch
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage, SystemMessage

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ──────────────────────────────
# 1) Key & Pfade
# ──────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
BACKEND    = BASE_DIR.parent / "backend"
VECTOR_DIR = BACKEND / "vector_db"
PDF_PATH   = BACKEND / "docs" / "mhb_wiing_BSc_de_aktuell.pdf"
def resolve_json_path() -> Path | None:
    # 1) ENV gewinnt
    env_p = os.getenv("RAG_JSON_PATH")
    if env_p and Path(env_p).exists():
        return Path(env_p)

    # 2) gängige Kandidaten
    candidates = [
        BACKEND / "TestText.json",           # backend/TestText.json
        BACKEND / "docs" / "TestText.json",  # backend/docs/TestText.json
        BASE_DIR.parent / "TestText.json",   # Projektroot/TestText.json
        BASE_DIR / "TestText.json",          # rag_pipeline/TestText.json
        Path.cwd() / "TestText.json",        # aktuelles Arbeitsverzeichnis
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

JSON_PATH = resolve_json_path()
if JSON_PATH:
    sys.stderr.write(f"[rag] TestText.json: {JSON_PATH}\n")
else:
    sys.stderr.write("[rag] Warnung: TestText.json nicht gefunden – RAG läuft nur mit PDF.\n")


load_dotenv(BACKEND / ".env")
api_key = os.getenv("OPENAI_API_KEY") or sys.exit("OPENAI_API_KEY fehlt")

# ──────────────────────────────
# 2) CLI-Argumente (Frage + Verlauf + Modus)
# ──────────────────────────────
if len(sys.argv) < 2:
    sys.exit("Frage fehlt!")

question_raw = sys.argv[1]
history_raw  = sys.argv[2] if len(sys.argv) > 2 else "[]"
mode         = sys.argv[3] if len(sys.argv) > 3 else "interview"

def convert_history(msgs: list[dict]):
    """{role, content} → [HumanMessage|AIMessage|SystemMessage]"""
    out = []
    for m in msgs:
        role    = m.get("role")
        content = m.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
    return out

try:
    history_dicts = json.loads(history_raw)
except json.JSONDecodeError:
    history_dicts = []

full_history = convert_history(history_dicts)

def keep_last_n_ai_turns(history, n=8):
    """Letzte n Bot-Antworten + davorstehende User-Fragen behalten."""
    keep_idx = set(); ai_seen = 0
    for i in range(len(history) - 1, -1, -1):
        m = history[i]
        if isinstance(m, AIMessage):
            if ai_seen >= n: continue
            ai_seen += 1; keep_idx.add(i)
            if i > 0 and isinstance(history[i - 1], HumanMessage):
                keep_idx.add(i - 1)
    return [history[i] for i in range(len(history)) if i in keep_idx]

def history_to_tuples(history):
    """[Human/AI/System] → [(human_text, ai_text), ...]"""
    pairs = []; last_h = None
    for m in history:
        if isinstance(m, HumanMessage): last_h = m.content
        elif isinstance(m, AIMessage) and last_h is not None:
            pairs.append((last_h, m.content)); last_h = None
    return pairs

def extract_candidate_set(history_dicts: list[dict]) -> list[str]:
    """Rückwärts die jüngste Bot-Liste (≥3 Punkte) als Kandidatensatz nehmen."""
    ai_seen = 0
    for m in reversed(history_dicts):
        if m.get("role") != "assistant": continue
        ai_seen += 1
        if ai_seen > 6: break
        text = m.get("content", ""); items = []
        for ln in text.splitlines():
            if re.match(r"^\s*(?:\d+[\.)]|[-•–])\s+", ln):
                item = re.sub(r"^\s*(?:\d+[\.)]|[-•–])\s+", "", ln)
                item = re.sub(r"\s*\(.*?LP.*?\)", "", item, flags=re.I)
                item = item.strip(" –-:;")
                if item: items.append(item)
        if len(items) >= 3:
            seen, out = set(), []
            for it in items:
                if it not in seen:
                    seen.add(it); out.append(it)
                if len(out) >= 20: break
            return out
    return []

def last_assistant_text(history_dicts: list[dict]) -> str:
    for m in reversed(history_dicts):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""

# Finale History + Kandidaten
trimmed_msgs   = keep_last_n_ai_turns(full_history, n=8)
chat_history   = history_to_tuples(trimmed_msgs)
candidate_list = extract_candidate_set(history_dicts)
candidate_text = "\n".join(candidate_list)
last_ai_text   = last_assistant_text(history_dicts)

def format_chat_history(pairs: list[tuple[str, str]]) -> str:
    """Transkript für den Condenser."""
    blocks = [f"User: {h}\nAssistant: {a}" for h, a in pairs[-12:]]
    return "\n\n".join(blocks)

# ──────────────────────────────
# 2a) Ontologie & Synonyme
# ──────────────────────────────
HAUPTFAECHER = {
    "betriebswirtschaftslehre": ["bwl", "management", "marketing", "controlling",
                                 "finanzierung", "finance", "rechnungswesen",
                                 "produktion", "wirtschaftsinformatik", "logistik",
                                 "hr", "strategie", "organisation"],
    "volkswirtschaftslehre": ["vwl", "ökonomie", "economics", "wirtschaftspolitik",
                              "makroökonomie", "mikroökonomie"],
    "informatik": ["computer science", "programmierung", "software",
                   "java", "ki", "künstliche intelligenz", "ai",
                   "security", "datenbanken", "internet computing"],
    "operations research": ["or", "optimierung", "operations research",
                            "supply chain", "netzwerke", "nichtlineare optimierung"],
    "ingenieurwissenschaften": ["ingenieurwesen", "ing", "maschinenbau", "mechatronik",
                                "elektrotechnik", "fahrzeug", "werkstoff",
                                "produktionstechnik", "mikrosystemtechnik", "bahnsystemtechnik"],
    "mathematik": ["mathe", "analysis", "lineare algebra", "differentialgleichungen"],
    "statistik": ["ökonometrie", "wahrscheinlichkeit", "regression"],
    "wahlpflichtbereich": ["wahlpflicht", "seminar", "teamprojekt", "recht", "soziologie"]
}

NORM_REPLACEMENTS = [
    (r"\bvorlesung(en)?\b", "Teilleistung"),
    (r"\bveranstaltung(en)?\b", "Teilleistung"),
    (r"\bkurs(e)?\b", "Teilleistung"),
    (r"\bprüf(ung|ungen)\b", "Teilleistung"),
    (r"\bmodul(e)?\b", "Modul"),
    (r"\bzusta?ndig(e|er|keit)?\b", "Verantwortung"),
    (r"\bprof(\.|essor(in)?)(en)?\b", "Verantwortung"),
    (r"\bhauptfach\b", "Bereich"),
]

def _normalize_synonyms(text: str) -> str:
    out = text
    for pat, repl in NORM_REPLACEMENTS:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out

def _detect_bereich(text: str) -> str | None:
    t = text.lower()
    for canonical, aliases in HAUPTFAECHER.items():
        if canonical in t:
            return canonical
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", t):
                return canonical
    m = re.search(r"bereich\s+([a-zäöüß\- ]{3,})", t)
    if m:
        cand = m.group(1).strip()
        for canonical, aliases in HAUPTFAECHER.items():
            if canonical in cand:
                return canonical
            if any(cand.startswith(a) or a in cand for a in aliases):
                return canonical
    return None

def enrich_question_with_ontology(user_q: str) -> str:
    q = _normalize_synonyms(user_q)
    bereich = _detect_bereich(q)
    if bereich:
        q += (
            f"\n\n[Hinweis: Mit 'Bereich' ist das Hauptfach '{bereich}' gemeint. "
            f"Hierarchie: Bereich/Hauptfach → Module → Teilleistungen. "
            f"Suche daher bevorzugt Module/Teilleistungen aus '{bereich}'.]"
        )
    else:
        if re.search(r"\bbereich\b", q.lower()):
            q += (
                "\n\n[Hinweis: 'Bereich' bedeutet hier 'Hauptfach' (BWL, VWL, Informatik, "
                "Operations Research, Ingenieurwissenschaften; zusätzlich Mathematik, Statistik, Wahlpflichtbereich). "
                "Hierarchie: Bereich/Hauptfach → Module → Teilleistungen.]"
            )
    q += "\n\n[Synonyme: Teilleistung≈Vorlesung/Kurs/Veranstaltung; Verantwortung≈zuständige Person/Professor/in; Bereich≈Hauptfach/Fach.]"
    return q

SYNONYMS = {
    "Teilleistung": [
        "Vorlesung", "Vorlesungen", "Lehrveranstaltung", "Lehrveranstaltungen",
        "Fach", "Fächer", "Seminar", "Seminare", "Übung", "Übungen",
        "Praktikum", "Praktika", "Labor", "Laborpraktikum"
    ],
    "Verantwortung": [
        "zuständige Person", "zuständig", "verantwortliche Person",
        "verantwortlicher Professor", "verantwortliche Professorin",
        "Dozent", "Dozentin", "Professor", "Professorin", "Prof."
    ],
}
synonyms_text = "\n".join(f"{k}: {', '.join(v)}" for k, v in SYNONYMS.items())

TL_SYNONYMS_RGX   = re.compile(r"\b(Vorlesung(?:en)?|Lehrveranstaltung(?:en)?|Fächer|Fach(?!hoch)|Seminar(?:e)?|Übung(?:en)?|Praktik(?:um|a)|Labor(?:praktikum)?)\b", re.I)
RESP_SYNONYMS_RGX = re.compile(r"\b(zuständig(?:e|er|en)?(?: Person)?|verantwortlich(?:e|er|en)?|Professor(?:in)?|Prof\.)\b", re.I)

def annotate_synonyms(q: str) -> str:
    t = q
    if TL_SYNONYMS_RGX.search(t) and not re.search(r"Teilleistung", t, re.I):
        t += " (gemeint: Teilleistungen)"
    if RESP_SYNONYMS_RGX.search(t) and not re.search(r"verantwort|zuständig", t, re.I):
        t += " (gemeint: verantwortliche Person / Verantwortung)"
    return t

# Frage anreichern
question = enrich_question_with_ontology(question_raw)
annotated_question = annotate_synonyms(question)

# ──────────────────────────────
# 3) Vector-DB + Retriever
# ──────────────────────────────
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024,
    openai_api_key=api_key,
)
vectordb = Chroma(persist_directory=str(VECTOR_DIR), embedding_function=embeddings)
dense_retriever = vectordb.as_retriever(search_kwargs={"k": 12})

# 3b) BM25 (+ PDF + JSON-Metadaten)
loader = PyPDFLoader(str(PDF_PATH))
bm25_docs = loader.load()
if JSON_PATH and JSON_PATH.exists():
    try:
        meta_json = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        from langchain.docstore.document import Document
        bm25_docs.extend(Document(page_content=e["text"], metadata=e) for e in meta_json)
    except Exception as e:
        sys.stderr.write(f"[rag] Fehler beim Laden von TestText.json: {e}\n")

bm25 = BM25Retriever.from_documents(bm25_docs); bm25.k = 20

# 3c) Self-Query (Metadaten nutzbar)
metadata_field_info = [
    AttributeInfo(name="ects_lp",        type="float",  description="Leistungspunkte"),
    AttributeInfo(name="responsibility", type="string", description="Verantwortlicher Dozent"),
]
self_query = SelfQueryRetriever.from_llm(
    ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0),
    vectordb,
    "Modul- und Teilleistungsbeschreibungen",
    metadata_field_info,
    verbose=False,
)

# 3d) Hybrid + Cross-Encoder-Reranker
hybrid = EnsembleRetriever(
    retrievers=[bm25, dense_retriever, self_query],
    weights=[0.25, 0.45, 0.30],
)
cross_encoder = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-large",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
)
reranker  = CrossEncoderReranker(model=cross_encoder, top_n=6)
retriever = ContextualCompressionRetriever(
    base_retriever=hybrid,
    base_compressor=reranker,
)

# ──────────────────────────────
# 4) Prompts & Chain
# ──────────────────────────────
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Du bist Studienberater des B.Sc.-Wirtschaftsingenieurwesens.\n"
     "Behandle folgende Synonyme als äquivalent:\n{synonyms}\n\n"
     "Begriffslogik:\n"
     "• 'Bereich' bedeutet 'Hauptfach' (BWL, VWL, Informatik, Operations Research, Ingenieurwissenschaften; "
     "  zusätzlich Mathematik, Statistik, Wahlpflichtbereich). Hierarchie: Bereich/Hauptfach → Module → Teilleistungen.\n"
     "• Wenn explizit nach Teilleistungen gefragt wird, nenne AUSSCHLIESSLICH Teilleistungen (IDs 'T-'); vermeide Module ('M-').\n"
     "• LP-Schwellen strikt beachten: 'mehr als X' ⇒ > X; 'mindestens X' ⇒ ≥ X; 'genau X' ⇒ == X.\n"
     "Zitiere nur Fakten aus dem Kontext. Wenn nichts belegbar ist, antworte exakt: 'Ich weiß es nicht.'"),
    ("user",
     "Kandidatensatz (optional):\n{candidate_set}\n\n"
     "Kontext:\n{context}\n\nFrage: {question}"),
]).partial(candidate_set=candidate_text, synonyms=synonyms_text)

condense_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Formuliere die letzte Nutzerfrage zu einer eigenständigen, präzisen Frage in deutscher Sprache um.\n"
     "Behandle folgende Synonyme als äquivalent:\n{synonyms}\n\n"
     "Nutze strikt den Gesprächsverlauf, um Referenzen wie 'diese'/'davon' aufzulösen und ALLE genannten Einschränkungen zu bewahren "
     "(z.B. Bereich/Hauptfach, zuvor genannte Fächer/Teilleistungen, LP-Schwellen, Semester, Verantwortungen).\n"
     "Nutze, falls angegeben, den Kandidatensatz als einzig zulässige Auswahlmenge.\n"
     "Füge KEINE neuen Annahmen hinzu. Antworte NUR mit der umformulierten Frage."),
    ("human",
     "Gesprächsverlauf (gekürzt):\n{chat_history}\n\n"
     "Kandidatensatz (optional):\n{candidate_set}\n\n"
     "Letzte Frage:\n{question}")
]).partial(candidate_set=candidate_text, synonyms=synonyms_text)

# LLMs
gen_llm  = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0, timeout=60, max_retries=2)
expl_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0, timeout=30, max_retries=1)

conv_chain = ConversationalRetrievalChain.from_llm(
    llm                         = gen_llm,
    retriever                   = retriever,
    verbose                     = False,
    return_source_documents     = True,
    combine_docs_chain_kwargs   = {"prompt": answer_prompt},
    condense_question_prompt    = condense_prompt,
    return_generated_question   = True,
    get_chat_history            = lambda pairs: "\n\n".join(f"User: {h}\nAssistant: {a}" for h, a in pairs[-12:]),
)

def safe_print(obj):
    print(json.dumps(obj, ensure_ascii=False)); sys.stdout.flush()

# ──────────────────────────────
# 5) Chain-Invoke (Q/A) ─────────────────────────────────
# ───────────────────────────────────────────────────────
try:
    chain_out = conv_chain.invoke({
        "question":     annotate_synonyms(enrich_question_with_ontology(question_raw)),
        "chat_history": history_to_tuples(keep_last_n_ai_turns(full_history, n=8)),
    })

    answer_txt = chain_out.get("answer", "") or "Ich weiß es nicht."
    gen_q      = chain_out.get("generated_question", "")
    raw_docs   = chain_out.get("source_documents", []) or []

    # Quellen deduplizieren & kappen
    seen = set(); clean_docs = []
    for d in raw_docs:
        src = d.metadata.get("source", d.metadata.get("title", "meta"))
        pg  = d.metadata.get("page", "–")
        key = (src, pg)
        if key in seen: continue
        seen.add(key); clean_docs.append(d)
    raw_docs = clean_docs[:4]

    result = {
        "answer": answer_txt,
        "generated_question": gen_q,
        "source_documents": [
            {"page": d.metadata.get("page", "–"),
             "source": d.metadata.get("source", d.metadata.get("title", "meta"))}
            for d in raw_docs
        ],
        "justification": "",
        "extracted_knowledge": []
    }

    # kurze Begründung (optional)
    try:
        context_excerpt = "\n\n".join(d.page_content for d in raw_docs)[:3000]
        explainer_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Du bist Tutor. Fasse in höchstens 3 Sätzen zusammen, warum die Antwort "
             "auf Basis des Kontexts plausibel ist. Keine neuen Infos hinzufügen."),
            ("user", "Frage: {question}\nAntwort: {answer}\nKontext:\n{context}"),
        ])
        just_msg = expl_llm.invoke(
            explainer_prompt.format(
                question=question_raw, answer=answer_txt, context=context_excerpt
            )
        )
        result["justification"] = (just_msg.content or "").strip()
    except Exception:
        result["justification"] = ""

    # ──────────────────────────────
    # 6) Interview-Extractor (nur im Interview-Modus)
    # ──────────────────────────────
    if mode.lower() == "interview":
        # Kategorien (aus deiner Liste, leicht normiert)
        CATEGORIES = [
            # Prüfung
            "Prüfung:Typ", "Prüfung:Lernstrategie", "Prüfung:Schwierigkeitsgrad",
            "Prüfung:Zeitkapazität", "Prüfung:Lerntipps", "Prüfung:Altklausuren",
            "Prüfung:Ähnlichkeit zu Übungsaufgaben",
            # Vorlesung
            "Vorlesung:Typ", "Vorlesung:Lohnenswert für Prüfung",
            "Vorlesung:Lernwert allgemein", "Vorlesung:Interaktivität",
            # Sonstiges
            "Kombinierfähigkeit", "Passende Berufsfelder", "Relevanz für die Zukunft",
            "Sympathie des Profs/Institut/Übungsleitung",
            "Lernmaterialien:Verfügbarkeit", "Lernmaterialien:Nützliche Foren",
            "Lernmaterialien:ILIAS sinnvoll"
        ]
        categories_text = "\n".join(f"- {c}" for c in CATEGORIES)

        # Hilfsdaten
        current_user_answer = question_raw.strip()
        previous_bot_msg    = last_ai_text
        candidate_blob      = candidate_text

        # Direkte ID-Erkennung (T-/M-)
        id_hits = re.findall(r"\b([TM]-[A-Z\-]+-\d{5,6})\b", current_user_answer)
        # Extractor-Prompt
        extractor_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Du extrahierst Expertenwissen aus einer Interview-Antwort.\n"
             "Aufgabe: Mappe die Antwort auf genau eine Ziel-Entität (Teilleistung 'T-…' oder Modul 'M-…'), "
             "bestimme eine passende Kategorie aus der vorgegebenen Liste und extrahiere den inhaltlichen Wert.\n"
             "Gib ein JSON-Array mit 0..n Records zurück. Jeder Record:\n"
             "{"
             "\"target_type\":\"teilleistung|modul\","
             "\"target_id\":\"T-…|M-…\","
             "\"category\":\"<EIN GENAUER LABEL AUS DER LISTE>\","
             "\"value\":\"<knapper Inhalt, 1–3 Sätze>\","
             "\"confidence\":0.0..1.0"
             "}\n\n"
             "Synonyme beachten: Teilleistung≈Vorlesung/Kurs/Veranstaltung, Verantwortung≈zuständige Person/Professor/in, Bereich≈Hauptfach/Fach.\n"
             "Wenn kein Ziel ermittelbar ist, gib ein leeres Array [] zurück."),
            ("user",
             "Kategorien:\n{categories}\n\n"
             "Kandidatensatz (zuletzt genannte Titel):\n{candidates}\n\n"
             "Letzte Bot-Frage / Kontext:\n{bot_msg}\n\n"
             "Nutzer-Antwort:\n{user_msg}\n\n"
             "ID-Hinweise (falls vorhanden): {id_hits}\n")
        ]).partial(categories=categories_text)

        extractor_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0, timeout=40)
        extr = extractor_llm.invoke(
            extractor_prompt.format(
                candidates=candidate_blob or "(keine)",
                bot_msg=previous_bot_msg or "(leer)",
                user_msg=current_user_answer,
                id_hits=", ".join(id_hits) if id_hits else "(keine)"
            )
        )
        try:
            extracted = json.loads(extr.content.strip())
            # Basic Validation + Clamp
            cleaned = []
            for r in extracted if isinstance(extracted, list) else []:
                ttype = str(r.get("target_type","")).lower()
                tid   = str(r.get("target_id","")).strip()
                cat   = str(r.get("category","")).strip()
                val   = str(r.get("value","")).strip()
                conf  = float(r.get("confidence", 0.0) or 0.0)
                if ttype not in ("teilleistung","modul"): continue
                if not re.match(r"^[TM]-[A-Z\-]+-\d{5,6}$", tid): continue
                if cat not in CATEGORIES: continue
                if not val: continue
                conf = max(0.0, min(1.0, conf))
                cleaned.append({
                    "target_type": ttype,
                    "target_id": tid,
                    "category": cat,
                    "value": val,
                    "confidence": conf
                })
            result["extracted_knowledge"] = cleaned
        except Exception:
            result["extracted_knowledge"] = []

    safe_print(result)

except Exception as e:
    safe_print({
        "answer": "Es gab ein technisches Problem bei der Auswertung. Bitte stelle deine letzte Frage erneut.",
        "generated_question": "",
        "source_documents": [],
        "justification": "",
        "extracted_knowledge": [],
        "error": f"{type(e).__name__}: {e}"
    })
