import React, { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useParams } from "react-router-dom";
import { askRag } from "../api/smartdoc";
import { getSessionId } from "../utils/session";

export default function DocumentQAPage() {
  const { id: doc_id } = useParams();
  const location = useLocation();

  const [q, setQ] = useState("");
  const [msgs, setMsgs] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [extractedText, setExtractedText] = useState(location.state?.extractedText || "");

  const bottomRef = useRef(null);
  const session_id = useMemo(() => getSessionId(), []);

  useEffect(() => {
    if (!extractedText && doc_id) {
      const all = JSON.parse(localStorage.getItem("uploadedDocs") || "[]");
      const doc = all.find((d) => d.id === doc_id);
      if (doc?.text) setExtractedText(doc.text);
    }
  }, [doc_id, extractedText]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs]);

  const send = async (text) => {
    if (!text.trim()) return;
    setMsgs((m) => [...m, { role: "user", content: text }]);
    setQ("");
    setBusy(true);
    setError(null);

    try {
      const data = await askRag({ question: text, doc_id, session_id, top_k: 5 });
      const refs = Array.isArray(data?.sources) && data.sources.length
        ? "\n\nReferences: " + data.sources.map((c) => `[${c.doc_id} #${c.seq}]`).join(", ")
        : "";
      setMsgs((m) => [...m, { role: "assistant", content: (data.answer || "No answer.") + refs }]);
    } catch (e) {
      console.error(e);
      setError(e?.response?.data?.error || e.message || "Request failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="qa-container">
      <div className="qa-left">
        <h2>📄 Document</h2>
        <pre className="qa-preformatted">{extractedText || "No document content available."}</pre>
      </div>

      <div className="qa-right">
        <h3>Ask anything about this document</h3>

        <div className="qa-suggestions">
          <button onClick={() => send("Give me a concise summary")}>Give me a concise summary</button>
          <button onClick={() => send("List key terms and definitions")}>List key terms and definitions</button>
          <button onClick={() => send("What policies are stated?")}>What policies are stated?</button>
        </div>

        <div style={{ flex: 1, overflowY: "auto", border: "1px solid #eee", borderRadius: 10, padding: 12, marginBottom: 12 }}>
          {msgs.map((m, i) => (
            <div key={i} style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 12, opacity: 0.6 }}>{m.role === "user" ? "You" : "Assistant"}</div>
              <div style={{ whiteSpace: "pre-wrap" }}>{m.content}</div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        {error && <div style={{ color: "crimson", marginBottom: 8 }}>{error}</div>}

        <div className="qa-input-section">
          <input
            type="text"
            placeholder="Type a question here…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !busy ? send(q) : null}
          />
          <button className="qa-send-btn" disabled={busy} onClick={() => send(q)}>➤</button>
        </div>
      </div>
    </div>
  );
}
