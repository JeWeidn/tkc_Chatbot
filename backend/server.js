// backend/server.js
import express from "express";
import dotenv  from "dotenv";
import cors    from "cors";
import fetch   from "node-fetch";
import fs      from "fs";
import path    from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process";

/* ───────── 1) Grund-Setup ───────── */
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

/* ───────── 2) klassische Chat-Route ───────── */
app.post("/api/chat", async (req, res) => {
  const { messages } = req.body;

  try {
    const openaiRes = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${API_KEY}`,
      },
      body: JSON.stringify({ model: "gpt-4", messages, temperature: 0.7 }),
    });

    const data = await openaiRes.json();

    if (data.choices?.[0]?.message?.content) {
      const conversation = {
        timestamp: new Date().toISOString(),
        messages,
        response: data.choices[0].message.content,
      };
      const existing = JSON.parse(
        fs.readFileSync(conversationsPath, "utf8")
      );

      if (existing.length === 0) {
        existing.push(conversation);               // erster Verlauf
      } else {
        // Letzte Konversation überschreiben (= gleiche Session)
        existing[existing.length - 1] = conversation;
      }
      fs.writeFileSync(conversationsPath, JSON.stringify(existing, null, 2), "utf8");
    }

    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify(data, null, 2));
  } catch (err) {
    console.error("Fehler:", err);
    res.status(500).json({ error: "Fehler beim Aufruf der OpenAI-API" });
  }
});

/* ───────── 3) RAG-Route ───────── */
app.post("/api/retrieve", (req, res) => {
  const { question, messages = [] } = req.body;          // ← erhält komplette Historie
  const ragScript = path.join(__dirname, "..", "rag_pipeline", "rag_query.py");

  const py = spawn("python", [ragScript, question], {
    cwd: path.join(__dirname, ".."),
  });

  let out = "";
  py.stdout.on("data", chunk => (out += chunk));
  py.stderr.on("data", err => console.error("RAG-Fehler:", err.toString()));

  py.on("close", () => {
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(out);                                        // Python liefert UTF-8

    /*  Konversation sichern  */
    try {
      const parsed = JSON.parse(out);                    // { result , source_documents }
      const botReply = {
        role:    "assistant",
        content: parsed.result,
        sources: parsed.source_documents,
      };

      const conversation = {
        timestamp: new Date().toISOString(),
        messages:  [...messages, botReply],              // ← User + Bot
      };

      const existing = JSON.parse(fs.readFileSync(conversationsPath, "utf8"));
      existing.push(conversation);
      fs.writeFileSync(conversationsPath, JSON.stringify(existing, null, 2), "utf8");
    } catch (e) {
      console.error("Persist-Fehler:", e);
    }
  });
});

/* ───────── 4) Admin-Route ───────── */
app.get("/api/conversations", (_req, res) => {
  try {
    const data = fs.readFileSync(conversationsPath, "utf8");
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(data);
  } catch {
    res.status(500).json({ error: "Fehler beim Lesen der Chatverläufe" });
  }
});

/* ───────── 5) Start ───────── */
app.listen(PORT, () => console.log(`Server läuft auf http://localhost:${PORT}`));
