import React, { useState } from "react";
import "./App.css";

function Chatbot() {
  const [userInput, setUserInput] = useState("");
  const [chatHistory, setChatHistory] = useState([]);

  /* ──────────────────────────────
     Nachricht senden
  ────────────────────────────── */
  const handleSend = async () => {
    if (!userInput.trim()) return;

    /* 1) User-Nachricht sofort anzeigen */
    const userMsg = { role: "user", content: userInput };
    const placeholder = {
      role: "assistant",
      content: "…",      // wird später ersetzt
      loading: true,
    };
    setChatHistory((prev) => [...prev, userMsg, placeholder]);
    setUserInput("");

    try {
      /* 2) RAG-Antwort holen */
       // Verlauf OHNE Platzhalter speichern
        const historyToSave = [...chatHistory, userMsg];   // ⇐ nur echte Messages
        const res = await fetch("http://localhost:5000/api/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
        question: userInput,
        messages: historyToSave      // ⇐ heißt jetzt messages
          }),
      });
      if (!res.ok) throw new Error("RAG-Endpoint nicht erreichbar");

      const { answer, sources = [] } = await res.json();

      /* Quellen mappen */
      const mappedSources = sources
        .filter((s) => s.page !== "–")            // leere raus
        .map((s, idx) => ({
          id:   idx,
          file: s.source.endsWith(".pdf") ? "Modulhandbuch" : s.source,
          page: Number(s.page) + 1,
        }));

      /* 3) Placeholder ersetzen */
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
      console.error("RAG-Fehler:", err);
      setChatHistory((prev) => {
        const newHist = [...prev];
        newHist[newHist.length - 1] = {
          role: "assistant",
          content: "Es gab ein Problem beim RAG-Abruf.",
          loading: false,
        };
        return newHist;
      });
    }
  };

  /* ──────────────────────────────
     Render
  ────────────────────────────── */
  return (
    <div className="App">
      <h2>Tacit Knowledge Chatbot</h2>

      {/* Chatverlauf */}
      <div className="chat-box">
        {chatHistory.map((msg, index) => (
          <div
            key={index}
            className={`message-row ${
              msg.role === "user" ? "user" : "assistant"
            }`}
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


              {/* Quellenanzeige */}
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
      </div>

      {/* Eingabe */}
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
          placeholder="Frage etwas..."
          rows={3}
        />
        <button type="submit">Senden</button>
      </form>
    </div>
  );
}

export default Chatbot;
