// backend/server.js – komplett aktualisiert
import express from "express";
import dotenv  from "dotenv";
import cors    from "cors";
import fetch   from "node-fetch";
import fs      from "fs";
import path    from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process";

/* ───────── 1) Grund‑Setup ───────── */
dotenv.config();
const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);
const PORT       = process.env.PORT || 5000;
const API_KEY    = process.env.OPENAI_API_KEY;

const conversationsPath = path.join(__dirname, "conversations.json");
if (!fs.existsSync(conversationsPath)) fs.writeFileSync(conversationsPath, "[]", "utf8");

const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

console.log("OpenAI-Key geladen:", API_KEY?.slice(0, 10) + "...");

/* ───────── 2) Standard‑OpenAI‑Route ───────── */
app.post("/api/chat", async (req, res) => {
  const { messages } = req.body;
  try {
    const rsp = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${API_KEY}`,
      },
      body: JSON.stringify({ model: "gpt-4", messages, temperature: 0.7 }),
    });

    const data   = await rsp.json();
    const answer = data.choices?.[0]?.message?.content ?? "";

    /* Konversation sichern */
    const log = JSON.parse(fs.readFileSync(conversationsPath, "utf8"));
    log.push({
      timestamp: new Date().toISOString(),
      messages,
      answer,
      justification: null,
      sources: [],
    });
    fs.writeFileSync(conversationsPath, JSON.stringify(log, null, 2));

    res.json({ answer });
  } catch (err) {
    console.error("Chat-API-Fehler:", err);
    res.status(500).json({ error: "OpenAI‑API fehlgeschlagen" });
  }
});

/* ───────── 3) RAG‑Route ───────── */
app.post("/api/retrieve", (req, res) => {
  const { question, messages = [] } = req.body;
  const ragScript = path.join(__dirname, "..", "rag_pipeline", "rag_query.py");

  const py = spawn("python", [ragScript, question], { cwd: path.join(__dirname, "..") });

  let stdout = "";
  py.stdout.on("data", chunk => (stdout += chunk));
  py.stderr.on("data", err => console.error("RAG‑Fehler:", err.toString()));

  py.on("close", () => {
    try {
      const parsed = JSON.parse(stdout);          // { result, source_documents, justification }
      const { result, source_documents, justification } = parsed;

      /* Antwort ans Frontend – keine justification */
      res.json({ answer: result, sources: source_documents });

      /* Verlauf loggen */
      const log = JSON.parse(fs.readFileSync(conversationsPath, "utf8"));
      log.push({
        timestamp: new Date().toISOString(),
        messages: [...messages, { role: "assistant", content: result }],
        justification,
        sources: source_documents,
      });
      fs.writeFileSync(conversationsPath, JSON.stringify(log, null, 2));
    } catch (e) {
      console.error("Persist‑Fehler / JSON‑Parse‑Error:", e);
      res.status(500).json({ error: "Interner Verarbeitungsfehler" });
    }
  });
});

/* ───────── 4) Admin‑Logs ───────── */
app.get("/api/conversations", (_req, res) => {
  try {
    const data = fs.readFileSync(conversationsPath, "utf8");
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(data);
  } catch {
    res.status(500).json({ error: "Fehler beim Lesen der Logs" });
  }
});

/* ───────── 5) Start ───────── */
app.listen(PORT, () => console.log(`Server läuft auf http://localhost:${PORT}`));
