import React, { useEffect, useState, useCallback } from "react";

function AdminView() {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [deleting, setDeleting] = useState({}); // { [sessionId]: true }

  const fetchConversations = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r   = await fetch("http://localhost:5000/api/conversations");
      const raw = await r.json(); // { sessions: { <sid>: {...} } }
      const map = raw?.sessions || {};

      const list = Object.entries(map).map(([sessionId, data]) => ({
        sessionId,
        ...data,
      }));

      list.sort((a, b) => {
        const ta = new Date(a?.started_at || a?.transcript?.[0]?.timestamp || 0).getTime() || 0;
        const tb = new Date(b?.started_at || b?.transcript?.[0]?.timestamp || 0).getTime() || 0;
        return tb - ta;
      });

      setSessions(list);
    } catch (e) {
      console.error("Fehler beim Laden:", e);
      setError("Konnte /api/conversations nicht laden.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchConversations();
  }, [fetchConversations]);

  const fmtTime = (iso) => {
    if (!iso) return "—";
    try {
      return new Date(iso).toLocaleString();
    } catch {
      return String(iso);
    }
  };

  const handleDelete = async (sessionId) => {
    const ok = window.confirm("Diesen Chatverlauf wirklich löschen?");
    if (!ok) return;

    setDeleting((d) => ({ ...d, [sessionId]: true }));
    setError(null);

    try {
      const resp = await fetch(
        `http://localhost:5000/api/conversations/${encodeURIComponent(sessionId)}`,
        { method: "DELETE" }
      );
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(txt || `HTTP ${resp.status}`);
      }
      // lokal entfernen
      setSessions((prev) => prev.filter((s) => s.sessionId !== sessionId));
    } catch (e) {
      console.error("Löschen fehlgeschlagen:", e);
      setError("Löschen fehlgeschlagen. Details in der Konsole.");
    } finally {
      setDeleting((d) => {
        const cp = { ...d };
        delete cp[sessionId];
        return cp;
      });
    }
  };

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: 20, fontFamily: "sans-serif" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
        <h2>🛠️ Admin: Chat-Sessions</h2>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={fetchConversations} disabled={loading}>
            {loading ? "Aktualisiere…" : "Aktualisieren"}
          </button>
        </div>
      </div>

      {error && (
        <div style={{ background: "#fee", border: "1px solid #f99", padding: 10, marginBottom: 16 }}>
          {error}
        </div>
      )}

      {sessions.length === 0 && !loading && (
        <div style={{ color: "#666" }}>Keine Sessions gefunden.</div>
      )}

      {sessions.map((s) => (
        <div
          key={s.sessionId}
          style={{
            border: "1px solid #ddd",
            borderRadius: 8,
            marginBottom: 20,
            overflow: "hidden",
          }}
        >
          {/* Kopf */}
          <div
            style={{
              background: "#f7f7f7",
              borderBottom: "1px solid #eee",
              padding: "10px 14px",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              gap: 12,
              flexWrap: "wrap",
            }}
          >
            <div style={{ fontWeight: 600 }}>
              Session: <span style={{ fontFamily: "monospace" }}>{s.sessionId}</span>
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <div style={{ color: "#555" }}>
                Start: {fmtTime(s.started_at)} &nbsp;|&nbsp; Modus:{" "}
                <strong>{s.mode || "—"}</strong> &nbsp;|&nbsp; Phase:{" "}
                <strong>{s.stage || "—"}</strong>
              </div>
              <button
                onClick={() => handleDelete(s.sessionId)}
                disabled={!!deleting[s.sessionId]}
                style={{
                  background: deleting[s.sessionId] ? "#aaa" : "#e72929",
                  color: "#fff",
                  border: "none",
                  borderRadius: 6,
                  padding: "6px 10px",
                  cursor: deleting[s.sessionId] ? "not-allowed" : "pointer",
                }}
                title="Diesen Chatverlauf löschen"
              >
                {deleting[s.sessionId] ? "Lösche…" : "Löschen"}
              </button>
            </div>
          </div>

          {/* General-Block */}
          <div style={{ padding: "10px 14px", borderBottom: "1px solid #eee" }}>
            <details open>
              <summary style={{ cursor: "pointer", fontWeight: 600 }}>Allgemeine Informationen</summary>
              <div style={{ marginTop: 8, display: "grid", gridTemplateColumns: "repeat(2, minmax(200px, 1fr))", gap: 8 }}>
                <div><strong>Semester:</strong> {s?.general?.semester ?? "—"}</div>
                <div><strong>Fortschritt (%):</strong> {s?.general?.progress_percent ?? "—"}</div>
                <div style={{ gridColumn: "1 / -1" }}>
                  <strong>Eindruck:</strong>
                  <div style={{ whiteSpace: "pre-wrap" }}>
                    {s?.general?.impression_text || "—"}
                  </div>
                </div>
                <div style={{ gridColumn: "1 / -1" }}>
                  <strong>Schwierige Bereiche:</strong>{" "}
                  {Array.isArray(s?.general?.difficult_areas) && s.general.difficult_areas.length
                    ? s.general.difficult_areas.join(", ")
                    : "—"}
                </div>
              </div>
              {!!s?.general_dynamic && (
                <div style={{ marginTop: 8, fontSize: 13, color: "#666" }}>
                  <em>Allgemein-Fragen gestellt:</em> {s.general_dynamic.asked ?? 0} / {s.general_dynamic.max ?? 0}
                </div>
              )}
            </details>
          </div>

          {/* Bereiche & Teilleistungen */}
          <div style={{ padding: "10px 14px", borderBottom: "1px solid #eee" }}>
            <details>
              <summary style={{ cursor: "pointer", fontWeight: 600 }}>
                Bereiche & Teilleistungen
              </summary>
              {s?.areas && Object.keys(s.areas).length > 0 ? (
                Object.entries(s.areas).map(([area, data]) => (
                  <div key={area} style={{ marginTop: 10, padding: 10, border: "1px dashed #ddd", borderRadius: 6 }}>
                    <div style={{ fontWeight: 600, marginBottom: 6 }}>
                      Bereich: {area}
                      <span style={{ color: "#777", fontWeight: 400 }}>
                        {" "}
                        (seit {fmtTime(data?.started_at)})
                      </span>
                    </div>
                    {data?.teilleistungen && Object.keys(data.teilleistungen).length > 0 ? (
                      Object.entries(data.teilleistungen).map(([tid, tdata]) => (
                        <div key={tid} style={{ margin: "6px 0 12px 0", paddingLeft: 10 }}>
                          <div style={{ fontFamily: "monospace" }}>{tid}</div>
                          <div style={{ fontWeight: 500 }}>{tdata?.title || "—"}</div>
                          {tdata?.collected && Object.keys(tdata.collected).length > 0 ? (
                            <div style={{ marginTop: 6 }}>
                              {Object.entries(tdata.collected).map(([cat, payload]) => (
                                <div
                                  key={cat}
                                  style={{ background: "#fafafa", border: "1px solid #eee", borderRadius: 6, padding: 8, marginTop: 6 }}
                                >
                                  <div style={{ fontSize: 13, color: "#555" }}>
                                    <strong>Kategorie:</strong> {cat}
                                  </div>
                                  <pre
                                    style={{
                                      margin: 0,
                                      whiteSpace: "pre-wrap",
                                      fontSize: 12,
                                      background: "transparent",
                                    }}
                                  >
{JSON.stringify(payload, null, 2)}
                                  </pre>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div style={{ color: "#888", fontSize: 13, marginTop: 6 }}>
                              (Noch keine strukturierten Angaben gesammelt)
                            </div>
                          )}
                        </div>
                      ))
                    ) : (
                      <div style={{ color: "#888", fontSize: 13 }}>(Keine Teilleistungen erfasst)</div>
                    )}
                  </div>
                ))
              ) : (
                <div style={{ color: "#888", fontSize: 13, marginTop: 8 }}>(Noch keine Bereiche erfasst)</div>
              )}
            </details>
          </div>

          {/* Transcript */}
          <div style={{ padding: "10px 14px" }}>
            <details open>
              <summary style={{ cursor: "pointer", fontWeight: 600 }}>
                Chatverlauf ({s?.transcript?.length || 0} Nachrichten)
              </summary>
              <div style={{ marginTop: 8 }}>
                {(s?.transcript || []).map((m, i) => (
                  <div
                    key={i}
                    style={{
                      padding: "6px 8px",
                      background: m.role === "assistant" ? "#eef6ff" : "#fff",
                      border: "1px solid #e6e6e6",
                      borderRadius: 6,
                      marginBottom: 6,
                    }}
                  >
                    <div style={{ fontSize: 12, color: "#666", marginBottom: 4 }}>
                      <strong>{m.role === "assistant" ? "Bot" : "User"}</strong>
                      <span style={{ marginLeft: 8 }}>{fmtTime(m.timestamp)}</span>
                    </div>
                    <div style={{ whiteSpace: "pre-wrap" }}>{m.content}</div>
                  </div>
                ))}
                {(!s?.transcript || s.transcript.length === 0) && (
                  <div style={{ color: "#888", fontSize: 13 }}>(Kein Verlauf)</div>
                )}
              </div>
            </details>
          </div>
        </div>
      ))}
    </div>
  );
}

export default AdminView;
