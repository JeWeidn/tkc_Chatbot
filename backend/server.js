// backend/server.js — LLM-first Interview + strukturierte Extraktion + JSON-LD/Turtle
// Änderungen (Kurz):
// - tl_search: Mehrfach-Erkennung -> wähle Kurs mit geringstem Informationsgehalt (Heuristik über TestText.json)
// - llmDetectEntities: found_tl_list + wrote_prob (+ wrote_hint) hinzugefügt
// - Hohe Schreibwahrscheinlichkeit -> direkte Phase-3-Frage mit kurzem Intro (kein extra "schon geschrieben?"-Prompt)
// - Sonst: eine kombinierte Frage ("Titel korrekt" + "geschrieben?") → llmCombinedTitleWrittenClassifier
// - in_tl > MAX_IN_TL_ROUNDS → Reset auf 0 + dynamische Wrap-up-Frage via llmPickPhaseQuestion
// - Evaluation: Zusammenfassung nur 1×, plus ausformulierte Abschnitte je Teilleistung (ohne Zählungen)
// - saveNewKnowledge: Merge statt stumpfes Anhängen (Duplikate/Fragmente werden reduziert)

import express from "express";
import dotenv  from "dotenv";
import cors    from "cors";
import fetch   from "node-fetch";
import fs      from "fs";
import path    from "path";
import { fileURLToPath } from "url";

dotenv.config();
const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// ── Basis-Konfiguration ───────────────────────────────────────────────────────
const PORT                   = process.env.PORT || 5000;
const OPENAI_API_KEY         = process.env.OPENAI_API_KEY;
const OPENAI_MODEL           = process.env.OPENAI_MODEL || "gpt-5-mini";
const OPENAI_FALLBACK_MODEL  = process.env.OPENAI_FALLBACK_MODEL || null;

// Rundenlimit für Phase 3 (Vertiefung)
const MAX_IN_TL_ROUNDS       = 6;

// ── Datei-/Pfad-Konfiguration ────────────────────────────────────────────────
const conversationsPath = path.join(__dirname, "conversations.json");
const catalogPath       = path.join(__dirname, "TestText.json");
const fragesetPath      = path.join(__dirname, "Fragenset.json");

const KG_JSONLD_PATH    = path.join(__dirname, "knowledge_graph.jsonld");
const KG_TURTLE_PATH    = path.join(__dirname, "knowledge.ttl");

// Verzeichnis für LLM-Traces (JSONL)
const TRACE_DIR         = path.join(__dirname, "traces");
if (!fs.existsSync(TRACE_DIR)) fs.mkdirSync(TRACE_DIR, { recursive: true });

// ── Fixer Kontext ────────────────────────────────────────────────────────────
const CONTEXT = {
  degree: "Wirtschaftsingenieurwesen",
  university: "Karlsruhe Institute of Technology (KIT)"
};

// ✨ Evaluation
const EVAL_QUESTIONS = [
  { id: "clarity",   prompt: "Wie klar waren die Fragen des Interviews?" },
  { id: "relevance", prompt: "Wie relevant waren die Fragen für dein Studium?" },
  { id: "pace",      prompt: "War das Tempo des Interviews angemessen?" },
  { id: "trust",     prompt: "Wie wohl hast du dich im Interview gefühlt?" },
  { id: "overall",   prompt: "Wie zufrieden bist du insgesamt?" }
];
const EVAL_SCHEMA = {
  items: EVAL_QUESTIONS,
  scale: { min: 1, max: 5, labels: { 1: "trifft gar nicht zu", 5: "trifft voll zu" } }
};

// ── Express ──────────────────────────────────────────────────────────────────
const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

if (!OPENAI_API_KEY) {
  console.warn("⚠️  OPENAI_API_KEY fehlt – LLM-Funktionen werden fehlschlagen.");
}

// ── Init-Dateien ─────────────────────────────────────────────────────────────
console.log("[paths] conversationsPath =", conversationsPath);
console.log("[paths] catalogPath       =", catalogPath);
console.log("[paths] fragesetPath      =", fragesetPath);
console.log("[paths] KG_JSONLD_PATH    =", KG_JSONLD_PATH);
console.log("[paths] KG_TURTLE_PATH    =", KG_TURTLE_PATH);

try {
  if (!fs.existsSync(KG_JSONLD_PATH)) fs.writeFileSync(KG_JSONLD_PATH, "[]", "utf8");
  if (!fs.existsSync(KG_TURTLE_PATH)) fs.writeFileSync(KG_TURTLE_PATH, "", "utf8");
} catch {}

// ── Store Utils ──────────────────────────────────────────────────────────────
function safeJsonRead(p, fallback) { try { return JSON.parse(fs.readFileSync(p,"utf8")); } catch { return fallback; } }
function ensureStoreFile() {
  if (!fs.existsSync(conversationsPath)) {
    fs.writeFileSync(conversationsPath, JSON.stringify({ sessions: {} }, null, 2), "utf8");
  }
}
function loadStore() { ensureStoreFile(); return safeJsonRead(conversationsPath, { sessions: {} }); }
function persistStore(store) {
  try { fs.writeFileSync(conversationsPath, JSON.stringify(store, null, 2), "utf8"); }
  catch (e) { console.error("[sessions] write error:", e); }
}

// ── Katalog / Kandidaten ─────────────────────────────────────────────────────
const catalogRaw  = safeJsonRead(catalogPath, []);
const fragesetRaw = safeJsonRead(fragesetPath, { phases: [] });

const teilleistungen = catalogRaw
  .filter(e => (e.teilleistung_id || /\[(T-[A-Z0-9\-]+)\]/i.test(e.title || "")))
  .map(e => {
    let tid = e.teilleistung_id;
    if (!tid) {
      const m = /\[(T-[A-Z0-9\-]+)\]/i.exec(e.title || "");
      if (m) tid = m[1];
    }
    return { id: tid || "", title: e.title || tid || "" };
  })
  .filter(e => e.id);

function norm(s){ return (s||"").toLowerCase()
  .replace(/ä/g,"ae").replace(/ö/g,"oe").replace(/ü/g,"ue").replace(/ß/g,"ss")
  .replace(/[^\p{L}\p{N}\s\-]/gu," ").replace(/\s+/g," ").trim(); }

function dice(a,b){
  const bg = s => { const t=norm(s); const r=[]; for(let i=0;i<t.length-1;i++) r.push(t.slice(i,i+2)); return r; };
  const A=bg(a), B=bg(b); if(!A.length||!B.length) return 0;
  let inter=0; const m=new Map(); A.forEach(x=>m.set(x,(m.get(x)||0)+1));
  B.forEach(x=>{ const c=m.get(x)||0; if(c>0){ inter++; m.set(x,c-1); }});
  return (2*inter)/(A.length+B.length);
}
function tokenJacc(a,b){
  const A=new Set(norm(a).split(" ").filter(Boolean));
  const B=new Set(norm(b).split(" ").filter(Boolean));
  let inter=0; A.forEach(x=>{ if(B.has(x)) inter++; });
  const uni=A.size+B.size-inter; return uni? inter/uni : 0;
}
function score(query, title){ return 0.6*dice(query,title)+0.4*tokenJacc(query,title); }
function candidates(query, k=10){
  const arr = teilleistungen.map(t => ({ id:t.id, title:t.title, score:score(query, t.title) }));
  arr.sort((a,b)=>b.score-a.score);
  return arr.slice(0,k);
}
function cleanTitle(title) {
  return String(title||"").replace(/\s*\[T-[A-Z0-9-]+\]\s*/g, "").trim();
}

// ── Sessions (+ Sanitizer) ───────────────────────────────────────────────────
function upsertSession(store, sessionId, mode="interview") {
  const s = store.sessions[sessionId] || (store.sessions[sessionId] = {
    mode,
    evaluation: { state: null, index: 0, answers: [] },
    started_at: new Date().toISOString(),
    stage: "await_semester_progress", // → general → tl_search → in_tl → wrap_up
    general: { semester: null, progress_percent: null },
    counters: { general_q: 0 },
    asked_log: [],
    current: {
      area: null,
      tl_id: null, tl_title: null,
      awaiting_written_confirm: false,
      awaiting_title_written_confirm: false, // NEU: kombinierte Bestätigung
      candidates: [], awaiting_candidate_choice: false,
      tl_facts: {},
      in_tl_rounds: 0,
      declinedWritten: [],
      last_confirm_tl: null,
      awaiting_tl_title_confirm: false, // legacy/backward compat
      pending_tl_candidate: null // { id, title }
    },
    flags: { llm_disabled: false, llm_disabled_reason: null },
    transcript: []
  });
  return s;
}
function sanitizeSessionShape(s) {
  s.stage    = s.stage || "await_semester_progress";
  s.general  = s.general || { semester: null, progress_percent: null };
  s.counters = s.counters || { general_q: 0 };
  s.asked_log = Array.isArray(s.asked_log) ? s.asked_log : [];
  s.transcript = Array.isArray(s.transcript) ? s.transcript : [];

  s.current = s.current || {};
  s.current.area   = (s.current.area ?? null);
  s.evaluation = s.evaluation || { state: null, index: 0, answers: [] };
  s.current.tl_id  = (s.current.tl_id ?? null);
  s.current.tl_title = (s.current.tl_title ?? null);

  s.current.awaiting_written_confirm  = !!s.current.awaiting_written_confirm;
  s.current.awaiting_title_written_confirm = !!s.current.awaiting_title_written_confirm;

  s.current.candidates = Array.isArray(s.current.candidates) ? s.current.candidates : [];
  s.current.awaiting_candidate_choice = !!s.current.awaiting_candidate_choice;

  s.current.tl_facts = (s.current.tl_facts && typeof s.current.tl_facts === "object") ? s.current.tl_facts : {};
  s.current.in_tl_rounds = Number.isFinite(s.current.in_tl_rounds) ? s.current.in_tl_rounds : 0;

  s.current.declinedWritten = Array.isArray(s.current.declinedWritten) ? s.current.declinedWritten : [];
  s.current.last_confirm_tl = s.current.last_confirm_tl || null;

  if (!("awaiting_tl_title_confirm" in s.current)) s.current.awaiting_tl_title_confirm = false;
  if (!("pending_tl_candidate" in s.current)) s.current.pending_tl_candidate = null;

  s.flags = s.flags || { llm_disabled: false, llm_disabled_reason: null };
}
function tlContextFromTranscript(s) {
  return s.transcript.map(m=>`${m.role.toUpperCase()}: ${m.content}`).join("\n");
}
function lastUserMessagesLines(transcript, k = 2) {
  const last = transcript.filter(m => m.role === "user").slice(-k);
  if (!last.length) return "(keine)";
  return last.map(m => `- ${m.content}`).join("\n");
}

// ── Transcript + Rationale ───────────────────────────────────────────────────
function pushTranscript(session, role, content, meta) {
  const last = session.transcript[session.transcript.length - 1];
  if (last && last.role === role && last.content === content) return;
  const entry = { role, content, timestamp: new Date().toISOString() };
  if (meta && typeof meta === "object") entry.meta = { ...meta };
  session.transcript.push(entry);
}
function rationaleTemplate(phase, code) {
  switch (code) {
    case "intro_greet": return "Einstieg mit Erhebung Basisdaten.";
    case "tl_search_title_only": return "Identifikation ausschließlich über exakten Titel.";
    case "tl_search_future_redirect": return "Zukunftsbezug erkannt → Rücklenkung auf bereits geschriebene Klausuren.";
    case "tl_search_remind": return "Bitte um präzise Titelangabe.";
    case "in_tl_first": return "Direkter Einstieg in die bestätigte Klausur.";
    case "wrap_up": return "Wechsel zu weiterer Teilleistung.";
    default: return "Schlüssige Fortschrittsfrage zur jeweiligen Phase.";
  }
}
function addAssistantTurn(session, text, code, ctx = {}) {
  const phase = session.stage;
  const rationale = ctx.rationale || rationaleTemplate(phase, code);
  pushTranscript(session, "assistant", text, { phase, rationale });
}

// ── LLM-Wrapper + Tracing ────────────────────────────────────────────────────
function logLLMTrace(sessionId, op, phase, messages, output) {
  try {
    const rec = { ts: new Date().toISOString(), sessionId, op, phase, messages, output };
    const file = path.join(TRACE_DIR, `${sessionId || "no_session"}.jsonl`);
    fs.appendFileSync(file, JSON.stringify(rec) + "\n", "utf8");
  } catch(e) { console.error("[trace] write error:", e); }
}
function supportsTemperature(model) {
  const m = String(model).toLowerCase();
  return !(m.startsWith("gpt-5") || m.startsWith("o1") || m.startsWith("o3") || m.startsWith("o4"));
}
async function openaiChatRaw({ model = OPENAI_MODEL, messages = [], temperature = 1, response_format = null, __traceMeta = null }) {
  const wantsJson = !!(response_format && response_format.type === "json_object");

  let _messages = Array.isArray(messages) ? [...messages] : [];

  if (wantsJson) {
    const hasJsonWord = _messages.some(m => String(m?.content || "").toLowerCase().includes("json"));
    if (!hasJsonWord) {
      _messages.unshift({
        role: "system",
        content: "Respond only with a valid json object. Output must be pure json (no prose)."
      });
    }
  }

  const payload = { model, messages: _messages };
  if (supportsTemperature(model)) payload.temperature = temperature;
  if (response_format) payload.response_format = response_format;

  const rsp = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: { Authorization: `Bearer ${OPENAI_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const text = await rsp.text();
  let json = null; try { json = JSON.parse(text); } catch {}

  if (!rsp.ok) {
    if (__traceMeta) logLLMTrace(__traceMeta.sessionId, __traceMeta.op, __traceMeta.phase, _messages, `HTTP ${rsp.status}: ${text}`);
    const err = new Error(`OpenAI HTTP ${rsp.status}: ${text}`);
    if (rsp.status === 429) {
      const code = json?.error?.code || json?.error?.type || "";
      if (String(code).includes("insufficient_quota")) err.isQuota = true;
      if (String(code).includes("rate")) err.isRateLimit = true;
    }
    throw err;
  }
  const data = json || {};
  const out = data.choices?.[0]?.message?.content ?? "";
  if (__traceMeta) logLLMTrace(__traceMeta.sessionId, __traceMeta.op, __traceMeta.phase, _messages, out);
  return out;
}

async function openaiChat(opts) {
  try { return await openaiChatRaw(opts); }
  catch (e) {
    if (!e.isQuota && OPENAI_FALLBACK_MODEL && opts.model !== OPENAI_FALLBACK_MODEL) {
      return await openaiChatRaw({ ...opts, model: OPENAI_FALLBACK_MODEL });
    }
    throw e;
  }
}

// ── Katalog-Helper (Erfolgskontrolle, Dozent) ────────────────────────────────
function getCatalogEntryById(tlId) {
  return safeJsonRead(catalogPath, []).find(e =>
    (e.teilleistung_id && e.teilleistung_id === tlId) || ((e.title||"").includes(`[${tlId}]`))
  ) || null;
}
function getErfolgskontrolleText(tlId) {
  const entry = getCatalogEntryById(tlId);
  if (!entry) return null;
  const txt = String(entry.text || "");
  const match = txt.match(/Erfolgskontrolle\(n\)\s*([\s\S]*?)(?:\n\s*[^\s.,]{1,20}\s*\n|$)/);
  return match ? match[1].trim() : null;
}
function getPrimaryInstructor(tlId) {
  const entry = getCatalogEntryById(tlId);
  if (!entry) return null;
  const cand = entry.dozent || entry.dozenten || entry.instructor || null;
  if (typeof cand === "string" && cand.trim()) return cand.trim();
  if (Array.isArray(cand) && cand.length) return String(cand[0]).trim();
  const t = String(entry.text||"");
  const m = t.match(/Dozent(?:in)?:\s*([^\n]+)/i);
  return m ? m[1].trim() : null;
}

// ── Heuristik „Informationsgehalt“ (für Auswahl bei mehreren Fächern) ────────
function factsSignalScore(facts = {}) {
  let sc = 0;
  if (facts.exam_type) sc++;
  if (Number.isFinite(facts.difficulty_1_5)) sc++;
  if (Number.isFinite(facts.prep_weeks)) sc++;
  if (Number.isFinite(facts.hours_per_week)) sc++;
  ["strategies","materials","pitfalls","tips"].forEach(k => {
    const v = facts[k];
    if (Array.isArray(v) && v.length) sc++; // 1 pro Kategorie reicht
  });
  return sc;
}
function infoScoreForTlId(tlId) {
  const entry = getCatalogEntryById(tlId);
  if (!entry) return 0; // Unbekannt = wenig Info → bevorzugen
  const nk = Array.isArray(entry.New_Knowledge) ? entry.New_Knowledge : [];
  let score = 0;
  nk.forEach(n => { score += factsSignalScore(n?.facts || {}); });
  score += Math.min(2, nk.length); // leichte Gewichtung: Häufigkeit
  const baseTxt = (entry.text || "").trim();
  if (baseTxt.length > 200) score += 1; // grober Textsignal
  return score;
}
function pickLeastKnownOfResolved(resolvedList) {
  if (!Array.isArray(resolvedList) || !resolvedList.length) return null;
  let best = null, bestScore = Infinity;
  for (const r of resolvedList) {
    const sc = infoScoreForTlId(r.id);
    if (sc < bestScore) { bestScore = sc; best = r; }
  }
  return best || resolvedList[0];
}

// ── LLM-Helfer (phasengerecht) ───────────────────────────────────────────────
function pickRandomFromPool(pool = [], askedLog = []) {
  const remaining = pool.filter(q => !askedLog.includes(q));
  const base = remaining.length ? remaining : pool;
  if (!base.length) return "Wie lief dein bisheriger Studienverlauf insgesamt?";
  return base[Math.floor(Math.random() * base.length)];
}

async function llmIntro(userText, _phase1Pool, _askedLog, hist, { sessionId=null, phase=null, recentUserContext=null } = {}) {
  const sys =
`PHASE 1 – Allgemeiner Einstieg.
Extrahiere:
{ "semester": number|null, "progress_percent": number|null }`;
  const messages = [
    { role:"system", content: sys },
    { role:"user", content:
`Nutzereingabe: ${userText}

Bekannt: Studiengang=${CONTEXT.degree}, Universität=${CONTEXT.university}
recent_user_messages:
${recentUserContext || "(keine)"}

Verlauf (kurz): ${hist}` }
  ];
  const out = await openaiChat({
    model: OPENAI_MODEL, temperature: 0.2, response_format: { type: "json_object" },
    messages, __traceMeta: { sessionId, op: "llmIntro", phase }
  });
  let p={}; try{ p=JSON.parse(out);}catch{}
  let { semester=null, progress_percent=null } = p || {};
  if (semester!=null && (!Number.isFinite(semester) || semester<1 || semester>20)) semester=null;
  if (progress_percent!=null) progress_percent=Math.max(0,Math.min(100,progress_percent|0));
  return { semester, progress_percent };
}

async function llmPickPhaseQuestion(phaseName, pool, askedLog, hist, { extraGuidance=null, sessionId=null, phase=null, recentUserContext=null } = {}) {
  const sys =
`Erzeuge EINE natürliche Frage (${phaseName}, deutsch).
Regeln:
- Passung zur Phase + letzten Nutzermeldungen.
- Keine Wiederholung (siehe already_asked).
- Gib JSON: { "question": string, "rationale": string }`;
  const messages = [
    { role:"system", content: sys },
    { role:"user", content:
`Bekannt: Studiengang=${CONTEXT.degree}, Universität=${CONTEXT.university}
recent_user_messages:
${recentUserContext || "(keine)"}

Fragen-Pool: ${JSON.stringify(pool)}
already_asked: ${JSON.stringify(askedLog)}
Kontext (jüngster Verlauf): ${hist}
zusätzliche Leitplanke: ${extraGuidance || "—"}
Gib NUR JSON zurück.` }
  ];
  const out = await openaiChat({ model: OPENAI_MODEL, temperature: 0.6, response_format: { type: "json_object" }, messages,
    __traceMeta: { sessionId, op: "llmPickPhaseQuestion", phase }});
  let o={}; try{ o=JSON.parse(out);}catch{}
  if (!o || !o.question) o = { question: pickRandomFromPool(pool, askedLog), rationale: "Pool-Fallback." };
  return { question: String(o.question).trim(), rationale: String(o.rationale||"").trim() };
}

// >>> llmDetectEntities (mit found_tl_list + wrote_prob) <<<
async function llmDetectEntities(
  userText,
  hist,
  { strict_current = false, recentUserContext = null, sessionId = null, phase = null } = {}
) {
  const sys =
`Erkenne in der aktuellen Nachricht erwähnte Teilleistungen (Klausuren/Vorlesungen).
Gib JSON:
{
  "found_area": string|null,
  "found_tl_text": string|null,
  "found_tl_list": string[]|null,
  "mentions_thesis": boolean,
  "thesis_topic": string|null,
  "temporal_hint": "past" | "future" | "mixed" | "unknown",
  "wrote_prob": number|null,
  "wrote_hint": "high" | "medium" | "low" | null
}

/*
Regeln:
- Nutze primary: found_tl_text ODER found_tl_list (nicht beides gleichzeitig, Ausnahme: found_tl_text bei genau einer Nennung).
- Lege wrote_prob konservativ fest; 0.85+ = "high".
- Wenn strict_current=true, orientiere dich ausschließlich an der aktuellen Nachricht; sonst nutze die letzten 1–2 Nutzermeldungen als leichte Hilfe.
*/`;

  const messages = [
    { role: "system", content: sys },
    { role: "user", content:
`Jetztige Nachricht: ${userText}
${strict_current ? "" : `Letzte 1–2 Nutzermeldungen:
${recentUserContext || "(keine)"} 
`}
Knappes Gesprächsfragment (nur zur Orientierung):
${hist}` }
  ];

  const out = await openaiChat({
    model: OPENAI_MODEL,
    temperature: 0,
    response_format: { type: "json_object" },
    messages,
    __traceMeta: { sessionId, op: "llmDetectEntities", phase }
  });

  let d = {};
  try { d = JSON.parse(out); } catch {}
  if (!Array.isArray(d.found_tl_list) || d.found_tl_list.length === 0) d.found_tl_list = null;

  if (d.found_area) d.found_area = String(d.found_area).toLowerCase();
  d.mentions_thesis = !!d.mentions_thesis;
  d.thesis_topic = d.thesis_topic || null;
  d.temporal_hint = d.temporal_hint || "unknown";
  if (d.wrote_prob != null && Number.isFinite(d.wrote_prob)) {
    d.wrote_prob = Math.max(0, Math.min(1, d.wrote_prob));
  } else d.wrote_prob = null;
  d.wrote_hint = d.wrote_hint || (d.wrote_prob!=null ? (d.wrote_prob>=0.85?"high":d.wrote_prob>=0.6?"medium":"low") : null);
  return d;
}

async function llmControlIntent(userText, hist, { sessionId=null, phase=null } = {}) {
  const sys = `Erkenne Abbruch vs. Weiter. Gib JSON: { "intent": "abort" | "continue" }`;
  const messages = [
    { role: "system", content: sys },
    { role: "user",   content: `Nachricht: ${userText}\nKontext: ${hist}` }
  ];
  const out = await openaiChat({
    model: OPENAI_MODEL, temperature: 0, response_format: { type: "json_object" },
    messages, __traceMeta: { sessionId, op: "llmControlIntent", phase }
  });
  let r={}; try{ r=JSON.parse(out);}catch{}
  const intent = (r.intent==="abort") ? "abort" : "continue";
  return { intent };
}

async function llmYesNo(userText, { sessionId=null, phase=null, op="llmYesNo" } = {}) {
  const sys = `Antworte als JSON: { "answer": "yes" | "no" | "unclear" }`;
  const messages = [{ role:"system", content: sys }, { role:"user", content: `Text: ${userText}` }];
  const out = await openaiChat({
    model: OPENAI_MODEL, temperature: 0, response_format: { type:"json_object" },
    messages, __traceMeta: { sessionId, op, phase }
  });
  let r={}; try{ r=JSON.parse(out);}catch{}
  return r?.answer || "unclear";
}

// Kombinierte Auswertung: Titel-Korrektheit + geschrieben?
async function llmCombinedTitleWrittenClassifier(userText, tlTitle, { sessionId=null, phase=null } = {}) {
  const sys =
`Interpretiere die Antwort auf: "Meintest du „${cleanTitle(tlTitle)}“ — und hast du diese Klausur bereits geschrieben?"
Gib JSON: { "title_match": "yes" | "no" | "unclear", "wrote": true | false | null }`;
  const messages = [
    { role:"system", content: sys },
    { role:"user",   content: `Antwort: ${userText}` }
  ];
  const out = await openaiChat({
    model: OPENAI_MODEL, temperature: 0, response_format: { type:"json_object" },
    messages, __traceMeta: { sessionId, op: "llmCombinedTitleWrittenClassifier", phase }
  });
  let r={}; try{ r=JSON.parse(out);}catch{}
  const title_match = (r?.title_match==="yes"||r?.title_match==="no") ? r.title_match : "unclear";
  const wrote = (r?.wrote===true) ? true : (r?.wrote===false ? false : null);
  return { title_match, wrote };
}

async function llmWrittenClassifier(userText, tlTitle, { sessionId=null, phase=null } = {}) {
  const sys =
`Beziehe dich auf die Frage: "Hattest du diese Klausur bereits geschrieben?"
Gib JSON: { "wrote_already": true | false | null }`;
  const messages = [
    { role:"system", content: sys },
    { role:"user",   content: `Klausur: ${tlTitle}\nAntwort: ${userText}` }
  ];
  const out = await openaiChat({
    model: OPENAI_MODEL, temperature: 0, response_format: { type:"json_object" },
    messages, __traceMeta: { sessionId, op: "llmWrittenClassifier", phase }
  });
  let r={}; try{ r=JSON.parse(out);}catch{}
  const v = (r.wrote_already===true) ? true : (r.wrote_already===false ? false : null);
  return v;
}

async function llmResolveTL(mention, cand, { sessionId=null, phase=null, recentUserContext=null } = {}) {
  const list = cand.map((c,i)=>`${i+1}) ${c.title} [${c.id}]`).join("\n");
  const sys =
`Wähle den besten Treffer.
Wenn unklar, gib need_clarify=true + kurze Rückfrage.
JSON:
{ "match_id": string|null, "match_title": string|null, "confidence": number, "need_clarify": boolean, "clarify_question": string|null }`;
  const messages = [
    { role:"system", content: sys },
    { role:"user",   content:
`Nutzerangabe: "${mention}"
Kandidaten:
${list}

recent_user_messages:
${recentUserContext || "(keine)"}

Gib nur JSON.` }
  ];
  const out = await openaiChat({
    model: OPENAI_MODEL, temperature: 0, response_format: { type:"json_object" },
    messages, __traceMeta: { sessionId, op: "llmResolveTL", phase }
  });
  let r={}; try{ r=JSON.parse(out);}catch{}
  r.confidence = Number.isFinite(r.confidence)? Math.max(0,Math.min(1,r.confidence)) : 0;
  if (r.match_id && !r.match_title) {
    const hit = cand.find(c => c.id===r.match_id);
    if (hit) r.match_title = hit.title;
  }
  return r;
}

async function llmPickPhase2IdentifyQuestion({ askedLog, hist, area=null, extraGuidance=null, sessionId=null, phase=null, recentUserContext=null }) {
  const sys = `PHASE 2 – Titelidentifikation (bereits geschriebene Klausur). 
EINE kurze Frage, ausschließlich zur Titel-Festlegung.
JSON: { "question": string, "rationale": string, "is_identification": true }`;
  const messages = [
    { role:"system", content: sys },
    { role:"user", content:
`already_asked: ${JSON.stringify(askedLog)}
aktuelles Fachgebiet: ${area || "—"}
recent_user_messages:
${recentUserContext || "(keine)"}

Kontext (kurz): ${hist}
zusätzliche Leitplanke: ${extraGuidance || "—"}
NUR JSON.` }
  ];
  const out = await openaiChat({
    model: OPENAI_MODEL,
    temperature: 0.4,
    response_format: { type: "json_object" },
    messages,
    __traceMeta: { sessionId, op: "llmPickPhase2IdentifyQuestion", phase }
  });

  let o={}; try { o = JSON.parse(out); } catch {}
  if (!o || typeof o !== "object" || !o.question) {
    o = { question: "Wie lautet der exakte Titel der bereits geschriebenen Klausur, über die du berichten möchtest?", rationale: "Fallback auf präzise Identifikationsfrage.", is_identification: true };
  }
  o.is_identification = true;
  return o;
}

async function llmExtractFacts(tlTitle, answerText, prevFacts={}, { sessionId=null, phase=null } = {}) {
  const sys =
`Extrahiere Fakten zu einer bereits geschriebenen Klausur:
{
 "exam_type": "schriftlich"|"mündlich"|null,
 "prep_weeks": number|null, "hours_per_week": number|null,
 "difficulty_1_5": number|null,
 "strategies": string[]|null, "materials": string[]|null,
 "pitfalls": string[]|null, "tips": string[]|null
}
JSON only.`;
  const messages = [
    { role:"system", content: sys },
    { role:"user",   content: `Klausur: ${tlTitle}\nBisher: ${JSON.stringify(prevFacts)}\nAntwort: ${answerText}` }
  ];
  const out = await openaiChat({
    model: OPENAI_MODEL, temperature: 0, response_format: { type:"json_object" },
    messages, __traceMeta: { sessionId, op: "llmExtractFacts", phase }
  });
  let f={}; try{ f=JSON.parse(out);}catch{}
  if (f.difficulty_1_5!=null && Number.isFinite(f.difficulty_1_5)) f.difficulty_1_5 = Math.max(1,Math.min(5,f.difficulty_1_5));
  ["prep_weeks","hours_per_week"].forEach(k => { if (f[k]!=null && !Number.isFinite(f[k])) f[k]=null; });
  return f;
}

async function llmNextTLQuestion({ tlTitle, tlId, phase3Pool, askedLog, tlFacts, erfolgskontrolle, hist, sessionId=null, phase=null, recentUserContext=null }) {
  const sys =
`PHASE 3 – Vertiefung (Vergangenheitsform).
Eine kurze präzise Frage. Keine Wiederholung bereits gestellter Fragen.
JSON: { "question": string, "rationale": string }`;
  const messages = [
    { role:"system", content: sys },
    { role:"user",   content:
`Klausur: ${tlTitle} [${tlId}]
recent_user_messages:
${recentUserContext || "(keine)"}

already_asked: ${JSON.stringify(askedLog)}
facts_so_far: ${JSON.stringify(tlFacts)}
erfolgskontrolle: ${erfolgskontrolle ? JSON.stringify(erfolgskontrolle) : "—"}
Fragen-Pool (Phase 3): ${JSON.stringify(phase3Pool)}
Kontext (jüngster Verlauf): ${hist}
NUR JSON.` }
  ];
  const out = await openaiChat({ model: OPENAI_MODEL, temperature: 0.6, response_format: { type: "json_object" }, messages,
    __traceMeta: { sessionId, op: "llmNextTLQuestion", phase }});
  let o={}; try{ o=JSON.parse(out);}catch{}
  if (!o || !o.question) o = { question: pickRandomFromPool(phase3Pool, askedLog), rationale: "Pool-Fallback." };
  return { question: String(o.question).trim(), rationale: String(o.rationale||"").trim() };
}

// ── Kandidaten-Antwort aus nummerierter Liste interpretieren ─────────────────
async function llmPickCandidateFromReply(cands, userText, { sessionId = null, phase = null } = {}) {
  if (!Array.isArray(cands) || !cands.length) return null;
  const list = cands.map(c => `${c.idx}) ${c.title} [${c.id}]`).join("\n");
  const sys = `Du bekommst eine nummerierte Kandidatenliste.
JSON (genau eines):
{ "decision": "pick", "idx": number }
{ "decision": "none" }
{ "decision": "free", "title": string }`;
  const messages = [
    { role: "system", content: sys },
    { role: "user", content: `Liste:\n${list}\nNutzeraussage: "${userText}"` }
  ];
  const out = await openaiChat({
    model: OPENAI_MODEL, temperature: 0,
    response_format: { type: "json_object" },
    messages, __traceMeta: { sessionId, op: "llmPickCandidateFromReply", phase }
  });
  let r = null; try { r = JSON.parse(out); } catch {}
  if (r && r.decision === "pick") {
    const idx = Number(r.idx);
    if (Number.isFinite(idx)) {
      const hit = cands.find(x => x.idx === idx);
      if (hit) return { ...hit, _decision: "pick" };
    }
  }
  if (r && r.decision === "none") return { _decision: "none" };
  if (r && r.decision === "free" && r.title) return { _decision: "free_title", title: String(r.title).trim() };
  if (r && r.idx !== undefined) {
    const idx = Number(r.idx);
    if (Number.isFinite(idx)) {
      const hit = cands.find(x => x.idx === idx);
      if (hit) return { ...hit, _decision: "pick" };
    }
  }
  return null;
}

// ── Fragenset-Pools ──────────────────────────────────────────────────────────
function getPhasePool(phaseNum) {
  const p = (fragesetRaw?.phases || []).find(x => x.phase === phaseNum);
  return Array.isArray(p?.questions) ? p.questions : [];
}

// ── Ontologie-Ausgabe (JSON-LD & Turtle) ─────────────────────────────────────
function toJSONLD(tlId, tlTitle, sessionId, facts) {
  return {
    "@context": {
      "ex": "http://example.org/wi-ontology#",
      "schema": "http://schema.org/",
      "Course": "schema:Course",
      "name": "schema:name",
      "examType": "ex:examType",
      "difficulty": "ex:difficulty",
      "prepWeeks": "ex:prepWeeks",
      "hoursPerWeek": "ex:hoursPerWeek",
      "strategy": "ex:strategy",
      "material": "ex:material",
      "pitfall": "ex:pitfall",
      "tip": "ex:tip",
      "evidence": "ex:evidence"
    },
    "@type": "Course",
    "@id": `http://example.org/wi-ontology#${tlId || norm(tlTitle).replace(/\s+/g,"_")}`,
    "name": tlTitle,
    "examType": facts.exam_type ?? null,
    "difficulty": facts.difficulty_1_5 ?? null,
    "prepWeeks": facts.prep_weeks ?? null,
    "hoursPerWeek": facts.hours_per_week ?? null,
    "strategy": facts.strategies ?? [],
    "material": facts.materials ?? [],
    "pitfall": facts.pitfalls ?? [],
    "tip": facts.tips ?? [],
    "evidence": sessionId
  };
}
function esc(str){ return String(str).replace(/\\/g,"\\\\").replace(/"/g,'\\"'); }
function toTurtle(tlId, tlTitle, sessionId, f) {
  const sid = tlId || norm(tlTitle).replace(/\s+/g,"_");
  const lines = [];
  lines.push(`@prefix ex: <http://example.org/wi-ontology#> .`);
  lines.push(`@prefix schema: <http://schema.org/> .`);
  lines.push(`ex:${sid} a schema:Course ;`);
  lines.push(`  schema:name "${esc(tlTitle)}" ;`);
  if (f.exam_type) lines.push(`  ex:examType "${esc(f.exam_type)}" ;`);
  if (Number.isFinite(f.difficulty_1_5)) lines.push(`  ex:difficulty ${f.difficulty_1_5} ;`);
  if (Number.isFinite(f.prep_weeks)) lines.push(`  ex:prepWeeks ${f.prep_weeks} ;`);
  if (Number.isFinite(f.hours_per_week)) lines.push(`  ex:hoursPerWeek ${f.hours_per_week} ;`);
  (f.strategies||[]).forEach(s => lines.push(`  ex:strategy "${esc(s)}" ;`));
  (f.materials||[]).forEach(s => lines.push(`  ex:material "${esc(s)}" ;`));
  (f.pitfalls||[]).forEach(s => lines.push(`  ex:pitfall "${esc(s)}" ;`));
  (f.tips||[]).forEach(s => lines.push(`  ex:tip "${esc(s)}" ;`));
  lines.push(`  ex:evidence "${esc(sessionId)}" .\n`);
  return lines.join("\n");
}
function appendToGlobalGraphs(jsonldDoc, ttlChunk) {
  const arr = fs.existsSync(KG_JSONLD_PATH) ? safeJsonRead(KG_JSONLD_PATH, []) : [];
  arr.push(jsonldDoc);
  fs.writeFileSync(KG_JSONLD_PATH, JSON.stringify(arr, null, 2), "utf8");
  fs.appendFileSync(KG_TURTLE_PATH, ttlChunk + "\n", "utf8");
}

// Merge-Strategie: gleiche Session & tlId → Fakten zusammenführen, nicht duplizieren
function mergeFacts(base = {}, add = {}) {
  const out = { ...base };
  const take = (k) => { if (add[k] != null) out[k] = add[k]; };
  ["exam_type","prep_weeks","hours_per_week","difficulty_1_5"].forEach(take);
  ["strategies","materials","pitfalls","tips"].forEach(k=>{
    const a = Array.isArray(base[k]) ? base[k] : [];
    const b = Array.isArray(add[k]) ? add[k] : [];
    const set = new Set([...a, ...b].filter(Boolean));
    out[k] = Array.from(set);
  });
  return out;
}
function saveNewKnowledge(tlId, sessionId, tlTitleClean, newFacts) {
  try {
    const raw = JSON.parse(fs.readFileSync(catalogPath, "utf8"));
    const idx = raw.findIndex(e =>
      (e.teilleistung_id && e.teilleistung_id === tlId) ||
      ((e.title||"").includes(`[${tlId}]`)) ||
      (cleanTitle(e.title || "").toLowerCase() === cleanTitle(tlTitleClean).toLowerCase())
    );
    if (idx<0) return false;

    const entry = raw[idx];
    entry.New_Knowledge = Array.isArray(entry.New_Knowledge) ? entry.New_Knowledge : [];

    // Prüfe, ob es bereits einen Eintrag derselben Session gibt → mergen
    const lastSameSessionIdx = [...entry.New_Knowledge].reverse()
      .findIndex(nk => nk?.sessionId === sessionId);
    if (lastSameSessionIdx >= 0) {
      const realIdx = entry.New_Knowledge.length - 1 - lastSameSessionIdx;
      const prev = entry.New_Knowledge[realIdx];
      const mergedFacts = mergeFacts(prev?.facts || {}, newFacts || {});
      const jsonldDoc = toJSONLD(tlId, tlTitleClean, sessionId, mergedFacts);
      const ttlChunk  = toTurtle(tlId, tlTitleClean, sessionId, mergedFacts);
      entry.New_Knowledge[realIdx] = {
        ...prev,
        timestamp: new Date().toISOString(),
        facts: mergedFacts,
        jsonld: jsonldDoc,
        ttl: ttlChunk
      };
      fs.writeFileSync(catalogPath, JSON.stringify(raw, null, 2), "utf8");
      appendToGlobalGraphs(jsonldDoc, ttlChunk);
      return true;
    }

    // Andernfalls neu anlegen
    const jsonldDoc = toJSONLD(tlId, tlTitleClean, sessionId, newFacts || {});
    const ttlChunk  = toTurtle(tlId, tlTitleClean, sessionId, newFacts || {});
    entry.New_Knowledge.push({
      sessionId,
      timestamp: new Date().toISOString(),
      facts: newFacts || {},
      jsonld: jsonldDoc,
      ttl: ttlChunk
    });

    raw[idx] = entry;
    fs.writeFileSync(catalogPath, JSON.stringify(raw, null, 2), "utf8");
    appendToGlobalGraphs(jsonldDoc, ttlChunk);
    return true;
  } catch(e) {
    console.error("[saveNewKnowledge] error:", e);
    return false;
  }
}

// ── Admin & Traces ───────────────────────────────────────────────────────────
app.get("/api/conversations", (_req, res) => {
  res.setHeader("Content-Type","application/json; charset=utf-8");
  res.end(fs.existsSync(conversationsPath)
    ? fs.readFileSync(conversationsPath,"utf8")
    : JSON.stringify({sessions:{}}));
});
app.delete("/api/conversations/:sessionId", (req, res) => {
  const store = loadStore();
  if (!store.sessions[req.params.sessionId]) {
    return res.status(404).json({error:"Session nicht gefunden"});
  }
  delete store.sessions[req.params.sessionId];
  persistStore(store);
  return res.status(204).end();
});
app.get("/api/traces/:sessionId", (req, res) => {
  const f = path.join(TRACE_DIR, `${req.params.sessionId}.jsonl`);
  if (!fs.existsSync(f)) return res.status(404).json({error:"Trace nicht gefunden"});
  res.setHeader("Content-Type","application/jsonl; charset=utf-8");
  res.end(fs.readFileSync(f, "utf8"));
});

// ── Evaluation-Helpers (Zusammenfassung & „schicke“ Wissenseinheiten) ───────
async function llmSummarizeTranscript(session, { sessionId=null } = {}) {
  const lastTurns = session.transcript.slice(-30).map(m => `${m.role}: ${m.content}`).join("\n");
  const sys = `Fasse den gesamten Chatverlauf auf Deutsch in 3–6 knappen Sätzen zusammen.
- Fokus: Welche Klausuren/Teilleistungen wurden besprochen? Welche Kernaussagen und Learnings hat der*die Nutzer*in geteilt?
- Schreibe als Fließtext (keine Liste, keine Zählungen, keine Überschriften).
- Kein Marketing, keine Wiederholungen, keine Metakommentare.`;
  const messages = [
    { role: "system", content: sys },
    { role: "user", content: lastTurns || "(leer)" }
  ];
  const out = await openaiChat({
    model: OPENAI_MODEL,
    temperature: 0.3,
    messages,
    __traceMeta: { sessionId, op: "llmSummarizeTranscript", phase: "evaluation" }
  });
  return String(out || "").trim();
}

// sammelt pro Session strukturierte Facts je TL
function collectSessionKnowledge(sessionId) {
  const raw = safeJsonRead(catalogPath, []);
  const byTlId = new Map(); // tlId -> { tlId, tlTitle, factsMerged }

  for (const entry of raw) {
    const tlId = entry?.teilleistung_id || ((entry?.title||"").match(/\[(T-[A-Z0-9\-]+)\]/)?.[1]) || null;
    const tlTitleClean = cleanTitle(entry?.title || "");
    const list = Array.isArray(entry?.New_Knowledge) ? entry.New_Knowledge : [];
    const mine = list.filter(nk => nk?.sessionId === sessionId);

    if (!mine.length || !tlId) continue;

    let merged = {};
    for (const nk of mine) merged = mergeFacts(merged, nk?.facts || {});
    const prev = byTlId.get(tlId);
    if (prev) {
      byTlId.set(tlId, { tlId, tlTitle: tlTitleClean, factsMerged: mergeFacts(prev.factsMerged, merged) });
    } else {
      byTlId.set(tlId, { tlId, tlTitle: tlTitleClean, factsMerged: merged });
    }
  }

  const items = [...byTlId.values()];
  return { structured: items };
}

// hübsche natürlichsprachliche Darstellung je TL
function naturalJoinDE(arr) {
  const a = (arr || []).filter(Boolean);
  if (a.length <= 1) return a.join("") || "";
  if (a.length === 2) return `${a[0]} und ${a[1]}`;
  return `${a.slice(0, -1).join(", ")} und ${a[a.length - 1]}`;
}
function mapDifficulty(de) {
  const m = {1:"sehr leicht",2:"leicht",3:"mittel",4:"anspruchsvoll",5:"sehr anspruchsvoll"};
  return m[de] || null;
}
function renderFactsNarrativeDE(title, tlId, f={}) {
  const parts = [];
  const diffTxt = Number.isFinite(f.difficulty_1_5) ? `${f.difficulty_1_5}/5 (${mapDifficulty(f.difficulty_1_5)})` : null;

  if (f.exam_type && diffTxt) {
    parts.push(`Die Prüfung war **${f.exam_type}** und wurde mit **${diffTxt}** eingeschätzt.`);
  } else if (f.exam_type) {
    parts.push(`Die Prüfung war **${f.exam_type}**.`);
  } else if (diffTxt) {
    parts.push(`Die Schwierigkeit wurde mit **${diffTxt}** eingeschätzt.`);
  }

  const wk = Number.isFinite(f.prep_weeks) ? `${f.prep_weeks} Woche${f.prep_weeks===1?"":"n"}` : null;
  const hpw = Number.isFinite(f.hours_per_week) ? `${f.hours_per_week} Std./Woche` : null;
  if (wk && hpw) parts.push(`Die Vorbereitung lag bei etwa **${wk}** bei **${hpw}**.`);
  else if (wk)   parts.push(`Die Vorbereitung umfasste etwa **${wk}**.`);
  else if (hpw)  parts.push(`Es wurden im Schnitt etwa **${hpw}** investiert.`);

  if (Array.isArray(f.strategies) && f.strategies.length) {
    parts.push(`Hilfreich waren insbesondere: ${naturalJoinDE(f.strategies)}.`);
  }
  if (Array.isArray(f.materials) && f.materials.length) {
    parts.push(`Als Materialien wurden genutzt: ${naturalJoinDE(f.materials)}.`);
  }
  if (Array.isArray(f.pitfalls) && f.pitfalls.length) {
    parts.push(`Typische Stolperfallen: ${naturalJoinDE(f.pitfalls)}.`);
  }
  if (Array.isArray(f.tips) && f.tips.length) {
    parts.push(`Praktische Tipps: ${naturalJoinDE(f.tips)}.`);
  }

  const body = parts.length ? parts.join(" ") : "Zu dieser Teilleistung wurden noch keine detaillierten Erkenntnisse formuliert.";
  return `### ${title} (${tlId})\n${body}`;
}
function renderKnowledgeMarkdown(items=[]) {
  if (!items.length) return "— (keine neuen Einträge gefunden)";
  return items.map(({ tlId, tlTitle, factsMerged }) => renderFactsNarrativeDE(tlTitle, tlId, factsMerged)).join("\n\n");
}

// --- Evaluation-Endpoints ---
app.post("/api/evaluation/start", async (req, res) => {
  const { sessionId } = req.body || {};
  if (!sessionId) return res.status(400).json({ error: "sessionId fehlt" });

  const store = loadStore();
  const s = upsertSession(store, sessionId, "interview");
  sanitizeSessionShape(s);

  s.evaluation = { state: "in_progress", index: 0, answers: [] };

  let summary = "";
  try { summary = await llmSummarizeTranscript(s, { sessionId }); } 
  catch { summary = "— (keine Zusammenfassung möglich)"; }

  const { structured: knowledgeItems } = collectSessionKnowledge(sessionId);
  const knowledgeMarkdown = renderKnowledgeMarkdown(knowledgeItems);

  // Chat-Text absichtlich schlank, damit die Zusammenfassung nicht doppelt erscheint
  const answer = "Evaluation gestartet.";

  addAssistantTurn(s, answer, "eval_start");
  persistStore(store);

  return res.json({
    answer,                         // nur kurzer Chat-Text
    eval_schema: EVAL_SCHEMA,
    summary,                        // 3–6 Sätze Fließtext
    knowledge_markdown: knowledgeMarkdown, // ausformulierte Abschnitte je TL (Markdown)
    new_knowledge: knowledgeItems,  // strukturiert (Kompatibilität)
    sessionId
  });
});

app.post("/api/evaluation/submit", (req, res) => {
  const { sessionId, ratings = {}, comments = "", corrections = "" } = req.body || {};
  if (!sessionId) return res.status(400).json({ error: "sessionId fehlt" });

  const invalid = Object.values(ratings).some(v => {
    const n = Number(v); return !Number.isFinite(n) || n < 1 || n > 5;
  });
  if (invalid) return res.status(400).json({ error: "ratings müssen 1..5 sein" });

  const store = loadStore();
  const s = upsertSession(store, sessionId, "interview");
  sanitizeSessionShape(s);

  s.evaluation = {
    state: "done",
    index: EVAL_QUESTIONS.length,
    answers: Object.entries(ratings).map(([id, value]) => ({ id, value: Number(value), ts: new Date().toISOString() })),
    comments: (comments || "").toString(),
    corrections: (corrections || "").toString()
  };

  try {
    const rec = {
      ts: new Date().toISOString(),
      sessionId,
      ratings: s.evaluation.answers,
      comments: s.evaluation.comments,
      corrections: s.evaluation.corrections
    };
    fs.appendFileSync(path.join(__dirname, "evaluations.jsonl"), JSON.stringify(rec) + "\n", "utf8");
  } catch (e) { console.warn("eval write warn:", e?.message); }

  const message = "Danke für dein Feedback! ✨";
  addAssistantTurn(s, message, "eval_done");
  persistStore(store);

  return res.json({ message, sessionId });
});

app.post("/api/eval/start", (req, res) => { req.url = "/api/evaluation/start"; app._router.handle(req, res, () => {}); });
app.post("/api/eval/answer", (req, res) => {
  const { sessionId, value, comment } = req.body || {};
  const ratings = { overall: Number(value) };
  req.body = { sessionId, ratings, comments: comment || "" };
  req.url = "/api/evaluation/submit";
  app._router.handle(req, res, () => {});
});

// ── Start/Reset Interview ────────────────────────────────────────────────────
function greetText(){
  return [
    "Hallo, ich bin dein Interview-Chatbot! Ich möchte dich zu deinem Studium befragen, um aus deinem wertvollen Erfahrungsschatz hilfreiches Wissen zu extrahieren.",
    "Zuerst stelle ich ein paar allgemeine Fragen und gehe danach konkreter auf einzelne Fächer/Klausuren ein. Am Ende bitte ich dich um eine kurze Evaluation.",
    "Zum Einstieg: In welchem Semester befindest du dich aktuell und wie weit bist du im Studium (in %)?"
  ].join(" ");
}
function hasIntroGreeting(session) {
  const g = greetText();
  return session.transcript.some(m => m.role === "assistant" && m.content === g);
}
function ensureIntroGreeting(session) { if (!hasIntroGreeting(session)) addAssistantTurn(session, greetText(), "intro_greet"); }

app.post("/api/interview/start", (req,res)=>{
  const { sessionId, mode="interview", force=false } = req.body||{};
  if (!sessionId) return res.status(400).json({error:"sessionId fehlt"});

  const store=loadStore();
  const s=upsertSession(store, sessionId, mode);
  sanitizeSessionShape(s);

  if (force) {
    s.stage="await_semester_progress";
    s.general={semester:null, progress_percent:null};
    s.counters={general_q:0};
    s.asked_log=[];
    s.current = {
      area: null,
      tl_id: null, tl_title: null,
      awaiting_written_confirm: false,
      awaiting_title_written_confirm: false,
      candidates: [], awaiting_candidate_choice: false,
      tl_facts: {}, in_tl_rounds: 0, declinedWritten: [], last_confirm_tl: null,
      awaiting_tl_title_confirm: false, pending_tl_candidate: null
    };

    s.flags = { llm_disabled:false, llm_disabled_reason:null };
    s.transcript=[];
  }

  ensureIntroGreeting(s);
  persistStore(store);
  res.json({ answer: greetText(), sources: [], sessionId });
});

app.post("/api/interview/reset", (req,res)=>{
  const { sessionId } = req.body||{};
  if(!sessionId) return res.status(400).json({error:"sessionId fehlt"});

  const store=loadStore();
  const s=upsertSession(store, sessionId, "interview");

  s.stage="await_semester_progress";
  s.general={semester:null, progress_percent:null};
  s.counters={general_q:0};
  s.asked_log=[];
  s.current={
    area:null,
    tl_id:null, tl_title:null,
    awaiting_written_confirm:false,
    awaiting_title_written_confirm:false,
    candidates:[], awaiting_candidate_choice:false,
    tl_facts:{}, in_tl_rounds:0, declinedWritten:[], last_confirm_tl:null,
    awaiting_tl_title_confirm:false, pending_tl_candidate:null
  };
  s.flags = { llm_disabled:false, llm_disabled_reason:null };
  s.transcript=[];

  sanitizeSessionShape(s);
  ensureIntroGreeting(s);

  persistStore(store);
  res.json({ answer: greetText(), sources: [], sessionId });
});

// ── Hauptlogik ───────────────────────────────────────────────────────────────
app.post("/api/retrieve", async (req,res)=>{
  const { question="", sessionId, mode="interview" } = req.body||{};
  if (!sessionId) return res.status(400).json({error:"sessionId fehlt"});
  const userText = String(question||"").trim();

  const store=loadStore();
  const s=upsertSession(store, sessionId, mode);
  sanitizeSessionShape(s);

  pushTranscript(s,"user",userText);
  persistStore(store);

  if (s.flags?.llm_disabled) {
    const msg = s.flags.llm_disabled_reason || "Ich kann gerade keine Antworten erzeugen (LLM deaktiviert). Bitte später erneut versuchen.";
    addAssistantTurn(s, msg, "system_quota");
    persistStore(store);
    return res.json({ answer: msg, sources: [], sessionId });
  }

  try {
    const histShort = tlContextFromTranscript(s);
    const recentUsersStr = lastUserMessagesLines(s.transcript, 2);
    const phase1 = getPhasePool(1);
    const phase3 = getPhasePool(3);
    const phase4 = getPhasePool(4);

    // 0) Intent
    const ctrl = await llmControlIntent(userText, histShort, { sessionId, phase: s.stage });
    if (ctrl.intent === "abort") {
      s.current = { ...s.current,
        tl_id:null, tl_title:null,
        awaiting_written_confirm:false,
        awaiting_title_written_confirm:false,
        awaiting_tl_title_confirm:false,
        pending_tl_candidate:null,
        candidates:[], awaiting_candidate_choice:false,
        in_tl_rounds:0, tl_facts:{}
      };
      s.stage = "tl_search";
      const { question:q, rationale } = await llmPickPhase2IdentifyQuestion({
        askedLog: s.asked_log, hist: histShort, area: s.current.area, sessionId, phase: s.stage, recentUserContext: recentUsersStr
      });
      s.asked_log.push(q);
      addAssistantTurn(s, q, "tl_search_title_only", { rationale });
      persistStore(store);
      return res.json({ answer: q, sources:[], sessionId });
    }

    // 1) Offene kandidatenauswahl?
    if (s.current.awaiting_candidate_choice && Array.isArray(s.current.candidates) && s.current.candidates.length) {
      const decision = await llmPickCandidateFromReply(s.current.candidates, userText, { sessionId, phase: s.stage });

      const combinedQuestion = (id, title) => {
        const instructor = getPrimaryInstructor(id);
        return `Nur zum Abgleich: Meintest du „${cleanTitle(title)}“${instructor?` (bei Dozent:in ${instructor})`:``} — und hast du diese Klausur bereits geschrieben?`;
      };

      if (decision?._decision === "pick") {
        const ask = combinedQuestion(decision.id, decision.title);
        s.current.pending_tl_candidate = { id: decision.id, title: decision.title };
        s.current.awaiting_title_written_confirm = true;
        s.current.awaiting_candidate_choice = false;
        s.current.candidates = [];
        addAssistantTurn(s, ask, "tl_search_confirm_title");
        persistStore(store);
        return res.json({ answer: ask, sources:[], sessionId });
      }

      if (decision?._decision === "none") {
        s.current.awaiting_candidate_choice = false;
        s.current.candidates = [];
        const { question:q2, rationale } = await llmPickPhase2IdentifyQuestion({
          askedLog: s.asked_log, hist: histShort, area: s.current.area, sessionId, phase: s.stage, recentUserContext: recentUsersStr
        });
        s.asked_log.push(q2);
        addAssistantTurn(s, q2, "tl_search_title_only", { rationale });
        persistStore(store);
        return res.json({ answer: q2, sources:[], sessionId });
      }

      if (decision?._decision === "free_title") {
        const mention = decision.title;
        const cand = candidates(mention, 10);
        const pick = await llmResolveTL(mention, cand, { sessionId, phase: s.stage, recentUserContext: recentUsersStr });

        if (pick.match_id && pick.confidence>=0.6) {
          const ask = combinedQuestion(pick.match_id, pick.match_title);
          s.current.pending_tl_candidate = { id: pick.match_id, title: pick.match_title };
          s.current.awaiting_title_written_confirm = true;
          s.current.awaiting_candidate_choice = false;
          s.current.candidates = [];
          addAssistantTurn(s, ask, "tl_search_confirm_title");
          persistStore(store);
          return res.json({ answer: ask, sources:[], sessionId });
        }

        if (pick.need_clarify && pick.clarify_question) {
          const top3 = cand.slice(0,3).map((c,i)=>({ idx:i+1, id:c.id, title:c.title }));
          s.current.candidates = top3;
          s.current.awaiting_candidate_choice = true;
          const list = top3.map(c=>`${c.idx}. ${cleanTitle(c.title)} (${c.id})`).join("\n");
          const msg = `${pick.clarify_question}\n\nZur Orientierung:\n${list}\n\nSag z. B. „die erste“ oder nenne den exakten Titel.`;
          addAssistantTurn(s, msg, "tl_search_remind");
          persistStore(store);
          return res.json({ answer: msg, sources:[], sessionId });
        }

        const { question:q2, rationale } = await llmPickPhase2IdentifyQuestion({
          askedLog: s.asked_log, hist: histShort, area: s.current.area, sessionId, phase: s.stage, recentUserContext: recentUsersStr
        });
        s.asked_log.push(q2);
        addAssistantTurn(s, q2, "tl_search_title_only", { rationale });
        persistStore(store);
        return res.json({ answer: q2, sources:[], sessionId });
      }

      const remind = "Wenn eine Option passte, sag bitte „die erste“ oder nenne den exakten Titel.";
      addAssistantTurn(s, remind, "tl_search_remind");
      persistStore(store);
      return res.json({ answer: remind, sources:[], sessionId });
    }

    // 2) Kombinierte Bestätigung (Titel + geschrieben?)
    if (s.current.awaiting_title_written_confirm && s.current.pending_tl_candidate) {
      const { title_match, wrote } = await llmCombinedTitleWrittenClassifier(userText, s.current.pending_tl_candidate.title, { sessionId, phase: s.stage });
      if (title_match === "yes") {
        if (wrote === true) {
          // Direkt in Phase 3 einsteigen → Runden & Fakten resetten
          s.current.tl_id    = s.current.pending_tl_candidate.id;
          s.current.tl_title = s.current.pending_tl_candidate.title;
          s.current.pending_tl_candidate = null;
          s.current.awaiting_title_written_confirm = false;
          s.stage = "in_tl";
          s.current.awaiting_written_confirm = false;
          s.current.last_confirm_tl = s.current.tl_id;
          s.current.in_tl_rounds = 0;       // ← Reset
          s.current.tl_facts = {};          // ← Reset Fakten für neue TL

          const erfolg = s.current.tl_id ? getErfolgskontrolleText(s.current.tl_id) : null;
          const { question:q, rationale } = await llmNextTLQuestion({
            tlTitle: cleanTitle(s.current.tl_title),
            tlId: s.current.tl_id,
            phase3Pool: getPhasePool(3),
            askedLog: s.asked_log,
            tlFacts: s.current.tl_facts,
            erfolgskontrolle: erfolg,
            hist: tlContextFromTranscript(s),
            sessionId, phase: s.stage,
            recentUserContext: recentUsersStr
          });
          s.asked_log.push(q);
          const intro = `Lass uns über „${cleanTitle(s.current.tl_title)}“ sprechen. ${q}`;
          addAssistantTurn(s, intro, "in_tl_first", { rationale: "Hohe Sicherheit: bereits geschrieben → direkter Einstieg." });
          persistStore(store);
          return res.json({ answer: intro, sources:[], sessionId });
        }
        if (wrote === false) {
          // Zurück zur Suche → Runden resetten
          if (s.current.last_confirm_tl && !s.current.declinedWritten.includes(s.current.last_confirm_tl)) {
            s.current.declinedWritten.push(s.current.last_confirm_tl);
          }
          s.stage="tl_search";
          s.current.tl_id=null; s.current.tl_title=null;
          s.current.awaiting_written_confirm=false;
          s.current.awaiting_title_written_confirm=false;
          s.current.last_confirm_tl=null;
          s.current.pending_tl_candidate=null;
          s.current.in_tl_rounds = 0;      // ← Reset

          const { question:q, rationale } = await llmPickPhase2IdentifyQuestion({
            askedLog: s.asked_log, hist: histShort, area: s.current.area, sessionId, phase: s.stage, recentUserContext: recentUsersStr
          });
          s.asked_log.push(q);
          addAssistantTurn(s, q, "tl_search_title_only", { rationale });
          persistStore(store);
          return res.json({ answer: q, sources:[], sessionId });
        }
        // title yes, wrote null → Nachfrage nur zum geschrieben-Status
        s.current.awaiting_written_confirm = true;
        s.current.awaiting_title_written_confirm = false;
        const reprompt = "Nur zur Sicherheit: Hattest du diese Klausur bereits geschrieben? (ja/nein)";
        addAssistantTurn(s, reprompt, "in_tl_first");
        persistStore(store);
        return res.json({ answer: reprompt, sources:[], sessionId });
      }
      if (title_match === "no") {
        // Wieder zur Identifikation
        s.current.pending_tl_candidate = null;
        s.current.awaiting_title_written_confirm = false;

        const { question:q2, rationale } = await llmPickPhase2IdentifyQuestion({
          askedLog: s.asked_log, hist: histShort, area: s.current.area, sessionId, phase: s.stage, recentUserContext: recentUsersStr
        });
        s.asked_log.push(q2);
        addAssistantTurn(s, q2, "tl_search_title_only", { rationale });
        persistStore(store);
        return res.json({ answer: q2, sources:[], sessionId });
      }
      // unklar → erneut kombinierte Nachfrage
      const instructor = getPrimaryInstructor(s.current.pending_tl_candidate.id);
      const ask = `Nur zum Abgleich: Meintest du „${cleanTitle(s.current.pending_tl_candidate.title)}“${instructor?` (bei Dozent:in ${instructor})`:``} — und hast du diese Klausur bereits geschrieben?`;
      addAssistantTurn(s, ask, "tl_search_confirm_title");
      persistStore(store);
      return res.json({ answer: ask, sources:[], sessionId });
    }

    /* A) await_semester_progress */
    if (s.stage==="await_semester_progress") {
      const { semester, progress_percent } =
        await llmIntro(userText, phase1, s.asked_log, histShort, { sessionId, phase: s.stage, recentUserContext: recentUsersStr });

      s.general.semester=semester;
      s.general.progress_percent=progress_percent;
      s.stage="general";
      s.counters.general_q = 0;
      s.current.awaiting_candidate_choice=false;
      s.current.candidates=[];

      let nextQObj = await llmPickPhaseQuestion("Allgemeine Fragen", phase1, s.asked_log, histShort, { sessionId, phase: s.stage, recentUserContext: recentUsersStr });
      if (!nextQObj?.question) {
        nextQObj = { question: pickRandomFromPool(phase1, s.asked_log), rationale: "Pool-Fallback." };
      }

      s.asked_log.push(nextQObj.question);
      addAssistantTurn(s, nextQObj.question, "intro_next_general", { rationale: nextQObj.rationale });
      persistStore(store);
      return res.json({ answer: nextQObj.question, sources:[], sessionId });
    }

    /* B) general */
    if (s.stage==="general") {
      const det = await llmDetectEntities(userText, histShort, {
        strict_current: false, recentUserContext: recentUsersStr, sessionId, phase: s.stage
      });
      if (det.found_area) s.current.area = det.found_area;

      // resolve helper
      const resolveMentions = async (mentions) => {
        const resolved = [];
        for (const m of mentions) {
          const cand = candidates(m, 10);
          const pick = await llmResolveTL(m, cand, { sessionId, phase: s.stage, recentUserContext: recentUsersStr });
          if (pick?.match_id && pick.confidence >= 0.6) resolved.push({ id: pick.match_id, title: pick.match_title, confidence: pick.confidence });
        }
        return resolved;
      };

      if (det.found_tl_list && det.found_tl_list.length > 1) {
        const resolved = await resolveMentions(det.found_tl_list);
        if (resolved.length) {
          const best = pickLeastKnownOfResolved(resolved);
          // Hohe Schreibwahrscheinlichkeit? Direkt in Phase 3
          if (det.wrote_prob != null && det.wrote_prob >= 0.85) {
            s.stage = "in_tl";
            s.current.tl_id = best.id; s.current.tl_title = best.title;
            s.current.awaiting_written_confirm = false;
            s.current.in_tl_rounds = 0;    // ← Reset
            s.current.tl_facts = {};       // ← Reset Fakten
            const erfolg = getErfolgskontrolleText(best.id);
            const { question:q, rationale } = await llmNextTLQuestion({
              tlTitle: cleanTitle(best.title), tlId: best.id, phase3Pool: phase3,
              askedLog: s.asked_log, tlFacts: s.current.tl_facts, erfolgskontrolle: erfolg,
              hist: tlContextFromTranscript(s), sessionId, phase: s.stage, recentUserContext: recentUsersStr
            });
            s.asked_log.push(q);
            const intro = `Lass uns über „${cleanTitle(best.title)}“ sprechen. ${q}`;
            addAssistantTurn(s, intro, "in_tl_first", { rationale: "Hohe Sicherheit aus Verlauf (geschrieben)." });
            persistStore(store);
            return res.json({ answer: intro, sources:[], sessionId });
          }

          // Sonst kombinierte Frage
          s.stage = "tl_search";
          s.current.pending_tl_candidate = { id: best.id, title: best.title };
          s.current.awaiting_title_written_confirm = true;
          const instructor = getPrimaryInstructor(best.id);
          const ask = `Nur zum Abgleich: Meintest du „${cleanTitle(best.title)}“${instructor?` (bei Dozent:in ${instructor})`:``} — und hast du diese Klausur bereits geschrieben?`;
          addAssistantTurn(s, ask, "tl_search_confirm_title");
          persistStore(store);
          return res.json({ answer: ask, sources:[], sessionId });
        }
      }

      if (det.found_tl_text || (det.found_tl_list && det.found_tl_list.length === 1)) {
        const mention = det.found_tl_text || det.found_tl_list[0];
        const cand = candidates(mention, 10);
        const pick = await llmResolveTL(mention, cand, { sessionId, phase: s.stage, recentUserContext: recentUsersStr });

        if (pick.match_id && pick.confidence>=0.6) {
          if (det.wrote_prob != null && det.wrote_prob >= 0.85) {
            // Direkt in Phase 3
            s.stage = "in_tl";
            s.current.tl_id = pick.match_id; s.current.tl_title = pick.match_title;
            s.current.awaiting_written_confirm = false;
            s.current.in_tl_rounds = 0;   // ← Reset
            s.current.tl_facts = {};      // ← Reset Fakten
            const erfolg = getErfolgskontrolleText(pick.match_id);
            const { question:q, rationale } = await llmNextTLQuestion({
              tlTitle: cleanTitle(pick.match_title), tlId: pick.match_id, phase3Pool: phase3,
              askedLog: s.asked_log, tlFacts: s.current.tl_facts, erfolgskontrolle: erfolg,
              hist: tlContextFromTranscript(s), sessionId, phase: s.stage, recentUserContext: recentUsersStr
            });
            s.asked_log.push(q);
            const intro = `Lass uns über „${cleanTitle(pick.match_title)}“ sprechen. ${q}`;
            addAssistantTurn(s, intro, "in_tl_first", { rationale: "Hohe Sicherheit aus Verlauf (geschrieben)." });
            persistStore(store);
            return res.json({ answer: intro, sources:[], sessionId });
          }

          // Kombiniert fragen
          s.stage = "tl_search";
          s.current.pending_tl_candidate = { id: pick.match_id, title: pick.match_title };
          s.current.awaiting_title_written_confirm = true;
          const instructor = getPrimaryInstructor(pick.match_id);
          const ask = `Nur zum Abgleich: Meintest du „${cleanTitle(pick.match_title)}“${instructor?` (bei Dozent:in ${instructor})`:``} — und hast du diese Klausur bereits geschrieben?`;
          addAssistantTurn(s, ask, "tl_search_confirm_title");
          persistStore(store);
          return res.json({ answer: ask, sources:[], sessionId });
        }
      }

      // Keine/unklare Teilleistung → allgemeine Frage
      const { question:q, rationale } = await llmPickPhaseQuestion("Allgemeine Fragen", phase1, s.asked_log, histShort, { sessionId, phase: s.stage, recentUserContext: recentUsersStr });
      const qFinal = q || pickRandomFromPool(phase1, s.asked_log);
      s.asked_log.push(qFinal);
      s.counters.general_q += 1;
      if (s.counters.general_q >= 2) s.stage = "tl_search";

      addAssistantTurn(s, qFinal, "general_more", { rationale: rationale || "Pool-Fallback." });
      persistStore(store);
      return res.json({ answer: qFinal, sources:[], sessionId });
    }

    /* C) tl_search */
    if (s.stage==="tl_search") {
      const det = await llmDetectEntities(userText, histShort, {
        strict_current: true,
        recentUserContext: recentUsersStr,
        sessionId,
        phase: s.stage
      });

      if (det?.found_area) s.current.area = det.found_area;

      if (det?.temporal_hint === "future") {
        const { question:q, rationale } = await llmPickPhase2IdentifyQuestion({
          askedLog: s.asked_log, hist: histShort, area: s.current.area,
          extraGuidance: "Bitte nur bereits geschriebene Klausuren nennen (Vergangenheit).",
          sessionId, phase: s.stage, recentUserContext: recentUsersStr
        });
        s.asked_log.push(q);
        addAssistantTurn(s, q, "tl_search_future_redirect", { rationale });
        persistStore(store);
        return res.json({ answer: q, sources:[], sessionId });
      }

      const resolveMentions = async (mentions) => {
        const resolved = [];
        for (const m of mentions) {
          const cand = candidates(m, 10);
          const pick = await llmResolveTL(m, cand, { sessionId, phase: s.stage, recentUserContext: recentUsersStr });
          if (pick?.match_id && pick.confidence >= 0.6) resolved.push({ id: pick.match_id, title: pick.match_title, confidence: pick.confidence });
        }
        return resolved;
      };

      // Mehrere Nennungen?
      if (det?.found_tl_list && det.found_tl_list.length > 1) {
        const resolved = await resolveMentions(det.found_tl_list);
        if (resolved.length) {
          const best = pickLeastKnownOfResolved(resolved);
          // Hohe Schreibwahrscheinlichkeit? Direkt Phase 3
          if (det.wrote_prob != null && det.wrote_prob >= 0.85) {
            s.stage = "in_tl";
            s.current.tl_id = best.id; s.current.tl_title = best.title;
            s.current.awaiting_written_confirm = false;
            s.current.in_tl_rounds = 0;   // ← Reset
            s.current.tl_facts = {};      // ← Reset Fakten
            const erfolg = getErfolgskontrolleText(best.id);
            const { question:q, rationale } = await llmNextTLQuestion({
              tlTitle: cleanTitle(best.title), tlId: best.id, phase3Pool: phase3,
              askedLog: s.asked_log, tlFacts: s.current.tl_facts, erfolgskontrolle: erfolg,
              hist: tlContextFromTranscript(s), sessionId, phase: s.stage, recentUserContext: recentUsersStr
            });
            s.asked_log.push(q);
            const intro = `Lass uns über „${cleanTitle(best.title)}“ sprechen. ${q}`;
            addAssistantTurn(s, intro, "in_tl_first", { rationale: "Hohe Sicherheit aus Verlauf (geschrieben)." });
            persistStore(store);
            return res.json({ answer: intro, sources:[], sessionId });
          }

          // sonst kombi-Frage
          s.current.pending_tl_candidate = { id: best.id, title: best.title };
          s.current.awaiting_title_written_confirm = true;
          const instructor = getPrimaryInstructor(best.id);
          const ask = `Nur zum Abgleich: Meintest du „${cleanTitle(best.title)}“${instructor?` (bei Dozent:in ${instructor})`:``} — und hast du diese Klausur bereits geschrieben?`;
          addAssistantTurn(s, ask, "tl_search_confirm_title");
          persistStore(store);
          return res.json({ answer: ask, sources:[], sessionId });
        }
      }

      // Einzelne/keine Nennung
      const mention = det?.found_tl_text || (det?.found_tl_list?.[0] ?? null);
      if (!mention) {
        const { question:q, rationale } = await llmPickPhase2IdentifyQuestion({
          askedLog: s.asked_log, hist: histShort, area: s.current.area, sessionId, phase: s.stage, recentUserContext: recentUsersStr
        });
        s.asked_log.push(q);
        addAssistantTurn(s, q, "tl_search_title_only", { rationale });
        persistStore(store);
        return res.json({ answer: q, sources:[], sessionId });
      }

      const cand = candidates(mention, 10);
      const pick = await llmResolveTL(mention, cand, { sessionId, phase: s.stage, recentUserContext: recentUsersStr });

      if (pick.match_id && pick.confidence>=0.6) {
        if (det.wrote_prob != null && det.wrote_prob >= 0.85) {
          // Direkt Phase 3
          s.stage = "in_tl";
          s.current.tl_id = pick.match_id; s.current.tl_title = pick.match_title;
          s.current.awaiting_written_confirm = false;
          s.current.in_tl_rounds = 0;   // ← Reset
          s.current.tl_facts = {};      // ← Reset Fakten
          const erfolg = getErfolgskontrolleText(pick.match_id);
          const { question:q, rationale } = await llmNextTLQuestion({
            tlTitle: cleanTitle(pick.match_title), tlId: pick.match_id, phase3Pool: phase3,
            askedLog: s.asked_log, tlFacts: s.current.tl_facts, erfolgskontrolle: erfolg,
            hist: tlContextFromTranscript(s), sessionId, phase: s.stage, recentUserContext: recentUsersStr
          });
          s.asked_log.push(q);
          const intro = `Lass uns über „${cleanTitle(pick.match_title)}“ sprechen. ${q}`;
          addAssistantTurn(s, intro, "in_tl_first", { rationale: "Hohe Sicherheit aus Verlauf (geschrieben)." });
          persistStore(store);
          return res.json({ answer: intro, sources:[], sessionId });
        }

        // kombinierte Frage
        s.current.pending_tl_candidate = { id: pick.match_id, title: pick.match_title };
        s.current.awaiting_title_written_confirm = true;
        const instructor = getPrimaryInstructor(pick.match_id);
        const ask = `Nur zum Abgleich: Meintest du „${cleanTitle(pick.match_title)}“${instructor?` (bei Dozent:in ${instructor})`:``} — und hast du diese Klausur bereits geschrieben?`;
        addAssistantTurn(s, ask, "tl_search_confirm_title");
        persistStore(store);
        return res.json({ answer: ask, sources:[], sessionId });
      }

      if (pick.need_clarify && pick.clarify_question) {
        const top3 = cand.slice(0,3).map((c,i)=>({ idx:i+1, id:c.id, title:c.title }));
        s.current.candidates = top3;
        s.current.awaiting_candidate_choice = true;
        const list = top3.map(c=>`${c.idx}. ${cleanTitle(c.title)} (${c.id})`).join("\n");
        const msg = `${pick.clarify_question}\n\nZur Orientierung:\n${list}\n\nSag z. B. „die erste“ oder nenne den exakten Titel.`;
        addAssistantTurn(s, msg, "tl_search_remind");
        persistStore(store);
        return res.json({ answer: msg, sources:[], sessionId });
      }

      const { question:q, rationale } = await llmPickPhase2IdentifyQuestion({
        askedLog: s.asked_log, hist: histShort, area: s.current.area, sessionId, phase: s.stage, recentUserContext: recentUsersStr
      });
      s.asked_log.push(q);
      addAssistantTurn(s, q, "tl_search_title_only", { rationale });
      persistStore(store);
      return res.json({ answer: q, sources:[], sessionId });
    }

    /* D) in_tl */
    if (s.stage==="in_tl") {
      s.current.in_tl_rounds = (s.current.in_tl_rounds||0) + 1;
      if (s.current.in_tl_rounds > MAX_IN_TL_ROUNDS) {
        s.current.in_tl_rounds = 0;   // ← Wichtig: Reset für die nächste TL
        s.stage="wrap_up";
        const { question:qWrap, rationale } = await llmPickPhaseQuestion(
          "Wechsel zu einer weiteren Teilleistung",
          phase4,
          s.asked_log,
          histShort,
          { extraGuidance: "Wenn möglich in Vergangenheitsform (bereits geschriebene Klausuren).", sessionId, phase: s.stage, recentUserContext: recentUsersStr }
        );
        const qFinal = qWrap || pickRandomFromPool(phase4, s.asked_log);
        s.asked_log.push(qFinal);
        addAssistantTurn(s, qFinal, "wrap_up", { rationale: rationale || "Pool-Fallback." });
        persistStore(store);
        return res.json({ answer: qFinal, sources:[], sessionId });
      }

      if (s.current.awaiting_written_confirm) {
        const wrote = await llmWrittenClassifier(userText, s.current.tl_title, { sessionId, phase: s.stage });
        if (wrote === true) {
          s.current.awaiting_written_confirm=false;
          const erfolg = s.current.tl_id ? getErfolgskontrolleText(s.current.tl_id) : null;
          const { question:q, rationale } = await llmNextTLQuestion({
            tlTitle: cleanTitle(s.current.tl_title),
            tlId: s.current.tl_id,
            phase3Pool: getPhasePool(3),
            askedLog: s.asked_log,
            tlFacts: s.current.tl_facts,
            erfolgskontrolle: erfolg,
            hist: tlContextFromTranscript(s),
            sessionId, phase: s.stage,
            recentUserContext: recentUsersStr
          });
          s.asked_log.push(q);
          addAssistantTurn(s, q, "in_tl_first", { rationale });
          persistStore(store);
          return res.json({ answer: q, sources:[], sessionId });
        }
        if (wrote === false) {
          if (s.current.last_confirm_tl && !s.current.declinedWritten.includes(s.current.last_confirm_tl)) {
            s.current.declinedWritten.push(s.current.last_confirm_tl);
          }
          s.stage="tl_search";
          s.current.tl_id=null;
          s.current.tl_title=null;
          s.current.awaiting_written_confirm=false;
          s.current.last_confirm_tl=null;
          s.current.in_tl_rounds = 0; // ← Reset

          const { question:q, rationale } = await llmPickPhase2IdentifyQuestion({
            askedLog: s.asked_log, hist: histShort, area: s.current.area, sessionId, phase: s.stage, recentUserContext: recentUsersStr
          });
          s.asked_log.push(q);
          addAssistantTurn(s, q, "tl_search_title_only", { rationale });
          persistStore(store);
          return res.json({ answer: q, sources:[], sessionId });
        }
        const reprompt = "Nur zur Sicherheit: Hattest du diese Klausur bereits geschrieben? (ja/nein)";
        addAssistantTurn(s, reprompt, "in_tl_first");
        persistStore(store);
        return res.json({ answer: reprompt, sources:[], sessionId });
      }

      // Fakten sammeln + speichern
      if (s.current.tl_title) {
        const add = await llmExtractFacts(cleanTitle(s.current.tl_title), userText, s.current.tl_facts, { sessionId, phase: s.stage });
        s.current.tl_facts = { ...(s.current.tl_facts||{}), ...add };
      }
      if (s.current.tl_title) {
        saveNewKnowledge(s.current.tl_id, sessionId, cleanTitle(s.current.tl_title), s.current.tl_facts);
      }

      const erfolg = s.current.tl_id ? getErfolgskontrolleText(s.current.tl_id) : null;
      const { question:qNext, rationale } = await llmNextTLQuestion({
        tlTitle: cleanTitle(s.current.tl_title),
        tlId: s.current.tl_id,
        phase3Pool: getPhasePool(3),
        askedLog: s.asked_log,
        tlFacts: s.current.tl_facts,
        erfolgskontrolle: erfolg,
        hist: histShort,
        sessionId, phase: s.stage,
        recentUserContext: recentUsersStr
      });

      const qFinal = qNext || pickRandomFromPool(getPhasePool(3), s.asked_log);
      const useQ = s.asked_log.includes(qFinal) ? pickRandomFromPool(getPhasePool(3), s.asked_log) : qFinal;
      s.asked_log.push(useQ);
      addAssistantTurn(s, useQ, "in_tl_next", { rationale: rationale || "Pool-Fallback." });
      persistStore(store);
      return res.json({ answer: useQ, sources:[], sessionId });
    }

    /* E) wrap_up */
    if (s.stage==="wrap_up") {
      const { question:q, rationale } = await llmPickPhaseQuestion(
        "Wechsel zu einer weiteren Teilleistung",
        getPhasePool(4),
        s.asked_log,
        histShort,
        { extraGuidance: "Wenn möglich, Vergangenheitsform (bereits geschriebene Klausuren).", sessionId, phase: s.stage, recentUserContext: recentUsersStr }
      );
      const qFinal = q || pickRandomFromPool(getPhasePool(4), s.asked_log);
      s.asked_log.push(qFinal);
      s.stage="tl_search";
      addAssistantTurn(s, qFinal, "wrap_up", { rationale: rationale || "Pool-Fallback." });
      persistStore(store);
      return res.json({ answer: qFinal, sources:[], sessionId });
    }

    const fb = "Lass uns mit einer bereits geschriebenen Klausur weitermachen: Wie lautet der exakte Titel?";
    addAssistantTurn(s, fb, "tl_search_title_only");
    persistStore(store);
    return res.json({ answer: fb, sources:[], sessionId });

  } catch (e) {
    console.error("/api/retrieve error:", e);

    const store2=loadStore();
    const s2=upsertSession(store2, sessionId, mode);
    sanitizeSessionShape(s2);

    if (e.isQuota) {
      s2.flags.llm_disabled = true;
      s2.flags.llm_disabled_reason =
        "Ich kann gerade keine Antworten erzeugen: Dein OpenAI-Kontingent (API) ist aufgebraucht oder Abrechnung ist nicht aktiv. Bitte im OpenAI-Dashboard prüfen.";
      const msg = s2.flags.llm_disabled_reason;
      addAssistantTurn(s2, msg, "system_quota");
      persistStore(store2);
      return res.json({ answer: msg, sources: [], sessionId });
    }

    if (e.isRateLimit) {
      const msg = "Ich wurde kurz gebremst (Rate Limit). Bitte sende dieselbe Nachricht in ein paar Sekunden erneut.";
      addAssistantTurn(s2, msg, "system_rate");
      persistStore(store2);
      return res.json({ answer: msg, sources: [], sessionId });
    }

    const fb = "Es gab ein technisches Problem. Bitte die letzte Nachricht kurz erneut senden.";
    addAssistantTurn(s2, fb, "system_error");
    persistStore(store2);
    return res.json({ answer: fb, sources:[], sessionId });
  }
});

// ── Serverstart ──────────────────────────────────────────────────────────────
console.log("LLM-Modell:", OPENAI_MODEL);
app.listen(PORT, ()=>console.log(`Server läuft auf http://localhost:${PORT}`));
