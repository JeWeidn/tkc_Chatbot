import React, { useEffect, useState } from "react";

function AdminView() {
  const [conversations, setConversations] = useState([]);

  /* ----------------------------------------------------------------
     Daten holen & Duplikate (Präfix-Verläufe) entfernen
  ----------------------------------------------------------------- */
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch("http://localhost:5000/api/conversations");
        let data = await r.json();                 // älteste → neueste
        data = data.reverse();                     // neueste zuerst

        /* ----------  Duplikate filtern  --------------------------- */
        const deduped = [];
        data.forEach((conv) => {
          // Prüfe, ob es bereits einen Verlauf gibt,
          // der mit denselben Nachrichten beginnt (Prefix) UND länger ist
          const isPrefixOfKept = deduped.some((kept) => {
            if (!kept.messages || !conv.messages) return false;
            if (conv.messages.length >= kept.messages.length) return false;

            // alle Messages von conv müssen in kept an gleicher Position stehen
            return conv.messages.every(
              (m, idx) =>
                idx < kept.messages.length &&
                kept.messages[idx].role === m.role &&
                kept.messages[idx].content === m.content
            );
          });

          if (!isPrefixOfKept) deduped.push(conv);
        });

        setConversations(deduped);
      } catch (err) {
        console.error("Fehler beim Laden:", err);
      }
    })();
  }, []);

  /* ----------------------------------------------------------------
     Render
  ----------------------------------------------------------------- */
  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: 20 }}>
      <h2>🛠️ Admin: Chatverläufe</h2>

      {conversations.map((conv, idx) => (
        <div
          key={idx}
          style={{
            border: "1px solid #ddd",
            marginBottom: 20,
            padding: 10,
          }}
        >
          <p>
            <strong>Datum:</strong>{" "}
            {new Date(conv.timestamp).toLocaleString()}
          </p>

          {conv.messages?.map((msg, i) => (
            <div key={i} style={{ marginBottom: 6 }}>
              <strong>{msg.role === "user" ? "User" : "Bot"}:</strong>{" "}
              {msg.content}
            </div>
          ))}

          {/* Fallback für uralte Einträge (nur response) */}
          {conv.response && !conv.messages?.some((m) => m.role === "assistant") && (
            <div style={{ marginBottom: 6 }}>
              <strong>Bot:</strong> {conv.response}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default AdminView;
