import React, { useEffect, useRef, useState } from "react";
import "./App.css";

function Chatbot() {
  /* ──────────────────────────────
     Basis-States
  ────────────────────────────── */
  const [userInput, setUserInput] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [booted, setBooted] = useState(false);
  const [error, setError] = useState("");

  /* ──────────────────────────────
     Modus: "interview" (Default) | "qa"
  ────────────────────────────── */
  const [mode, setMode] = useState(() => {
    const m = localStorage.getItem("mode");
    return m === "qa" ? "qa" : "interview";
  });
  const onModeChange = (m) => {
    setMode(m);
    localStorage.setItem("mode", m);
  };

  /* ──────────────────────────────
     Session-ID persistent
  ────────────────────────────── */
  const [sessionId, setSessionId] = useState(() => {
    const stored = localStorage.getItem("sessionId");
    if (stored) return stored;
    const id = "sess_" + Date.now() + "_" + Math.random().toString(36).slice(2, 8);
    localStorage.setItem("sessionId", id);
    return id;
  });

  /* ──────────────────────────────
     Evaluation-UI
     evalSchema: { items:[{id,prompt}], scale:{min,max,labels{1..5}} }
  ────────────────────────────── */
  const [evalSchema, setEvalSchema] = useState(null);
  const [evalAnswers, setEvalAnswers] = useState({});   // { [itemId]: 1..5 }
  const [evalComments, setEvalComments] = useState("");
  const [evalCorrections, setEvalCorrections] = useState("");

  const resetEvaluationUI = () => {
    setEvalSchema(null);
    setEvalAnswers({});
    setEvalComments("");
    setEvalCorrections("");
  };

  /* ──────────────────────────────
     Autoscroll
  ────────────────────────────── */
  const boxRef = useRef(null);
  useEffect(() => {
    if (boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [chatHistory, evalSchema]);

  /* ──────────────────────────────
     1) Begrüßung / Interviewstart
     (idempotent – Backend verhindert Duplikate,
      Frontend ergänzt nur, wenn Verlauf leer ist)
  ────────────────────────────── */
  const startInterview = async (force = false) => {
    try {
      setLoading(true);
      setError("");
      const r = await fetch("http://localhost:5000/api/interview/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId, mode: "interview", force }),
      });
      if (!r.ok) throw new Error(`Start-HTTP ${r.status}`);
      const data = await r.json();
      const greet = data?.answer || "Hallo! (Fallback)";

      // Nur dann ergänzen, wenn der Verlauf noch leer ist
      setChatHistory((prev) => (prev.length ? prev : [{ role: "assistant", content: greet }]));
      setBooted(true);
    } catch (e) {
      console.error("Interview-Start fehlgeschlagen:", e);
      setError("Interview-Start fehlgeschlagen. Bitte „Neu starten“ versuchen.");
    } finally {
      setLoading(false);
    }
  };

  // Beim Mount / Moduswechsel
  useEffect(() => {
    if (mode === "interview") {
      startInterview(false);
    } else {
      // Im QA-Modus kein automatischer Server-Start nötig — Button „Evaluation starten“
      // bleibt dennoch verfügbar (booted=true).
      setBooted(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, sessionId]);

  /* ──────────────────────────────
     2) Nachricht senden (Interview & Q&A)
  ────────────────────────────── */
  const handleSend = async () => {
    if (loading) return;
    if (!userInput.trim()) return;

    const nowQuestion = userInput.trim();

    // User-Turn + Platzhalter
    const userMsg = { role: "user", content: nowQuestion };
    const placeholder = { role: "assistant", content: "…", loading: true };
    setChatHistory((prev) => [...prev, userMsg, placeholder]);
    setUserInput("");
    setError("");
    setLoading(true);

    try {
      // Optional: Verlauf ohne Platzhalter (nur echte Nachrichten)
      const historyToSave = [...chatHistory, userMsg];

      const res = await fetch("http://localhost:5000/api/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: nowQuestion,
          messages: historyToSave, // Backend ignoriert es aktuell, stört aber nicht
          sessionId,
          mode,
        }),
      });

      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(`RAG/Interview-HTTP ${res.status}: ${txt || "Kein Text"}`);
      }

      const payload = await res.json();
      const answer = payload?.answer ?? "Ich weiß es nicht.";
      const sourcesRaw = Array.isArray(payload?.sources) ? payload.sources : [];

      // Quellen mappen (PDF → „Modulhandbuch“, sonst „Quelle“), Seiten +1
      const mappedSources = sourcesRaw
        .filter((s) => s.page !== "–")
        .map((s, idx) => ({
          id: idx,
          file: s.source && s.source.endsWith?.(".pdf") ? "Modulhandbuch" : (s.source || "Quelle"),
          page: Number(s.page) + 1,
        }));

      // Platzhalter ersetzen
      setChatHistory((prev) => {
        const newHist = [...prev];
        newHist[newHist.length - 1] = {
          role: "assistant",
          content: answer,
          sources: mappedSources,
          loading: false,
        };
        return newHist;
      });
    } catch (err) {
      console.error("Senden fehlgeschlagen:", err);
      setChatHistory((prev) => {
        const newHist = [...prev];
        newHist[newHist.length - 1] = {
          role: "assistant",
          content: "Es gab ein Problem bei der Verarbeitung.",
          loading: false,
        };
        return newHist;
      });
      setError("Senden fehlgeschlagen. Details in der Konsole.");
    } finally {
      setLoading(false);
    }
  };

  /* ──────────────────────────────
     3) Hard Reset (neue Session)
  ────────────────────────────── */
  const handleHardReset = async () => {
    try {
      setLoading(true);
      setError("");

      // neue Session-ID
      const newId = "sess_" + Date.now() + "_" + Math.random().toString(36).slice(2, 8);
      localStorage.setItem("sessionId", newId);
      setSessionId(newId);

      // UI zurücksetzen
      setChatHistory([]);
      setBooted(false);
      resetEvaluationUI();

      // Server neu booten (force)
      const r = await fetch("http://localhost:5000/api/interview/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: newId, mode: "interview", force: true }),
      });
      if (!r.ok) throw new Error(`Reset-HTTP ${r.status}`);

      const data = await r.json();
      const greet = data?.answer || "Hallo! (Fallback)";
      setChatHistory([{ role: "assistant", content: greet }]);
      setBooted(true);
    } catch (e) {
      console.error("Neu starten fehlgeschlagen:", e);
      setError("Neu starten fehlgeschlagen. Siehe Konsole.");
    } finally {
      setLoading(false);
    }
  };

  /* ──────────────────────────────
     4) Evaluation starten (jederzeit)
     - keine Stage-Prüfung
     - zeigt Summary + öffnet Formular
  ────────────────────────────── */
  const startEvaluation = async () => {
    try {
      setLoading(true);
      setError("");
      const r = await fetch("http://localhost:5000/api/evaluation/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId }),
      });
      if (!r.ok) {
        const txt = await r.text().catch(() => "");
        throw new Error(`Eval-Start-HTTP ${r.status}: ${txt || "Kein Text"}`);
      }
      const data = await r.json();

      let msgText = data?.answer || data?.message || "Evaluation gestartet.";

      // Optional: Falls du lieber lokal formatieren willst
      if (!data?.answer && (data?.summary || (data?.new_knowledge?.length > 0))) {
        const bullets = (data.new_knowledge || []).map(
          (it) => {
            const f = it.facts || {};
            const parts = [];
            if (f.exam_type) parts.push(`Typ=${f.exam_type}`);
            if (Number.isFinite(f.difficulty_1_5)) parts.push(`Schwierigkeit=${f.difficulty_1_5}/5`);
            const c = (a)=>Array.isArray(a)?a.length:0;
            const counts = [];
            if (c(f.strategies)) counts.push(`Strategien=${c(f.strategies)}`);
            if (c(f.materials)) counts.push(`Materialien=${c(f.materials)}`);
            if (c(f.pitfalls)) counts.push(`Stolperfallen=${c(f.pitfalls)}`);
            if (c(f.tips)) counts.push(`Tipps=${c(f.tips)}`);
            const tail = [...parts, ...counts].join("; ");
            return `• ${it.title} (${it.tl_id})${tail ? `: ${tail}` : ""}`;
          }
        ).join("\n");
        msgText =
          "Evaluation gestartet.\n\n" +
          "Kurze Zusammenfassung:\n" + (data.summary || "—") + "\n\n" +
          "Neue Wissenseinheiten (diese Session):\n" + (bullets || "— (keine neuen Einträge gefunden)");
      }

setChatHistory((prev) => [...prev, { role: "assistant", content: msgText }]);


setChatHistory((prev) => [...prev, { role: "assistant", content: msgText }]);


      // Formular-Schema setzen (falls keins kommt, bleibt Panel zu)
      if (data?.eval_schema) {
        setEvalSchema(data.eval_schema);
        setEvalAnswers({});
      } else {
        // Optional: Fallback-Form anzeigen, wenn Backend kein Schema liefert
        setEvalSchema({
          items: [
            { id: "clarity", prompt: "Die Fragen waren klar und verständlich." },
            { id: "usefulness", prompt: "Das Interview war für mich hilfreich." },
            { id: "pace", prompt: "Das Tempo war angenehm." },
          ],
          scale: { min: 1, max: 5, labels: { 1: "stimme nicht zu", 5: "stimme voll zu" } },
        });
        setEvalAnswers({});
      }
    } catch (e) {
      console.error("Evaluation-Start fehlgeschlagen:", e);
      setError("Evaluation-Start fehlgeschlagen.");
    } finally {
      setLoading(false);
    }
  };

  /* ──────────────────────────────
     5) Evaluation absenden
  ────────────────────────────── */
  const submitEvaluation = async () => {
    try {
      setLoading(true);
      setError("");

      const r = await fetch("http://localhost:5000/api/evaluation/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId,
          ratings: evalAnswers,
          comments: evalComments,
          corrections: evalCorrections,
        }),
      });
      if (!r.ok) {
        const txt = await r.text().catch(() => "");
        throw new Error(`Eval-Submit-HTTP ${r.status}: ${txt || "Kein Text"}`);
      }
      const data = await r.json();

      setChatHistory((prev) => [
        ...prev,
        { role: "assistant", content: data?.message || "Danke für dein Feedback!" },
      ]);
      resetEvaluationUI();
    } catch (e) {
      console.error("Evaluation-Submit fehlgeschlagen:", e);
      setError("Evaluation-Absenden fehlgeschlagen.");
    } finally {
      setLoading(false);
    }
  };

  /* ──────────────────────────────
     Render
  ────────────────────────────── */
  return (
    <div className="App">
      <div className="header">
        <h2>Tacit Knowledge Chatbot</h2>

        {/* Modus-Toggle + Reset + Evaluation */}
        <div className="mode-toggle" role="group" aria-label="Modus umschalten">
          <button
            type="button"
            className={`mode-btn ${mode === "interview" ? "active" : ""}`}
            onClick={() => onModeChange("interview")}
            title="Interviewer-Modus"
            disabled={loading}
          >
            🎤 Interview
          </button>
          <button
            type="button"
            className={`mode-btn ${mode === "qa" ? "active" : ""}`}
            onClick={() => onModeChange("qa")}
            title="Frage-Antwort-Modus"
            disabled={loading}
          >
            ❓ Q&A
          </button>
          <button
            type="button"
            className="mode-btn"
            onClick={handleHardReset}
            disabled={loading}
            title="Neue Session starten"
            style={{ marginLeft: 8 }}
          >
            🔁 Neu starten
          </button>
          <button
            type="button"
            className="mode-btn"
            onClick={startEvaluation}
            // 👉 Evaluation jederzeit möglich (nur gesperrt, solange gerade geladen wird
            //    oder der Client noch nicht gebootet ist)
            disabled={loading || !booted}
            title="Evaluation starten"
            style={{ marginLeft: 8 }}
          >
            ⭐ Evaluation starten
          </button>
        </div>
      </div>

      {error && (
        <div style={{ color: "#b00020", marginBottom: 8, fontSize: 14 }}>{error}</div>
      )}

      {/* Chatverlauf */}
      <div className="chat-box" ref={boxRef}>
        {chatHistory.map((msg, index) => (
          <div
            key={index}
            className={`message-row ${msg.role === "user" ? "user" : "assistant"}`}
          >
            <div className={`bubble ${msg.role}`}>
              {msg.role === "assistant" && <span className="bot-icon">🤖</span>}
              <span className="bubble-text">
                {msg.loading ? (
                  <em className="loading-dots">
                    <span>.</span>
                    <span>.</span>
                    <span>.</span>
                  </em>
                ) : (
                  msg.content
                )}
              </span>

              {/* Quellenanzeige (falls vorhanden) */}
              {msg.role === "assistant" && msg.sources?.length > 0 && (
                <div className="sources">
                  {msg.sources.map((s) => (
                    <span key={s.id}>
                      [{s.file}, S.&nbsp;{s.page}]
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Hinweis beim Booten */}
        {mode === "interview" && !booted && (
          <div style={{ color: "#666", fontSize: 14, marginTop: 8 }}>
            Initialisiere Interview…
          </div>
        )}
      </div>

      {/* Evaluation-Panel */}
      {evalSchema && (
        <div className="eval-panel" style={{ borderTop: "1px solid #ddd", paddingTop: 12, marginTop: 8 }}>
          <h4>Evaluation</h4>
          <p style={{ marginTop: 0, color: "#555" }}>
            Bitte bewerte jede Aussage von 1 („{evalSchema.scale?.labels?.[1] || "stimme nicht zu"}“) bis 5 („
            {evalSchema.scale?.labels?.[5] || "stimme voll zu"}“).
          </p>

          <div className="eval-items">
            {evalSchema.items.map((it) => (
              <div key={it.id} className="eval-item" style={{ marginBottom: 10 }}>
                <div style={{ marginBottom: 4 }}>{it.prompt}</div>
                <div role="radiogroup" aria-label={it.prompt} style={{ display: "flex", gap: 8 }}>
                  {[1, 2, 3, 4, 5].map((v) => (
                    <label key={v} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                      <input
                        type="radio"
                        name={`eval_${it.id}`}
                        value={v}
                        checked={Number(evalAnswers[it.id]) === v}
                        onChange={() => setEvalAnswers((prev) => ({ ...prev, [it.id]: v }))}
                        disabled={loading}
                      />
                      <span>{v}</span>
                    </label>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: 10 }}>
            <label style={{ display: "block", fontWeight: 500 }}>Anmerkungen (optional):</label>
            <textarea
              rows={3}
              value={evalComments}
              onChange={(e) => setEvalComments(e.target.value)}
              placeholder="Was lief gut? Was könnte besser werden?"
              disabled={loading}
              style={{ width: "100%" }}
            />
          </div>

          <div style={{ marginTop: 10 }}>
            <label style={{ display: "block", fontWeight: 500 }}>Korrekturen zu Wissenspunkten (optional):</label>
            <textarea
              rows={3}
              value={evalCorrections}
              onChange={(e) => setEvalCorrections(e.target.value)}
              placeholder="Wenn etwas aus der Zusammenfassung nicht stimmt, bitte hier kurz korrigieren."
              disabled={loading}
              style={{ width: "100%" }}
            />
          </div>

          <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
            <button type="button" onClick={submitEvaluation} disabled={loading}>
              {loading ? "Sende Bewertung…" : "Bewertung absenden"}
            </button>
            <button type="button" onClick={resetEvaluationUI} disabled={loading}>
              Abbrechen
            </button>
          </div>
        </div>
      )}

      {/* Eingabebereich */}
      <form
        className="input-area"
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
      >
        <textarea
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder={mode === "interview" ? "Kurze Antwort auf die Interviewfrage eingeben…" : "Frage etwas…"}
          rows={3}
          disabled={loading || (mode === "interview" && !booted)}
        />
        <button type="submit" disabled={loading || (mode === "interview" && !booted)}>
          {loading ? "Senden…" : "Senden"}
        </button>
      </form>
    </div>
  );
}

export default Chatbot;
