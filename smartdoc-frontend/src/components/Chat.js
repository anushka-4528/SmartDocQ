import React, { useState } from 'react';
import { askRag } from '../api/smartdoc';
import { getSessionId } from '../utils/session';

export default function Chat({ docId }) {
  const [q, setQ] = useState('');
  const [a, setA] = useState('');
  const [busy, setBusy] = useState(false);
  const session_id = getSessionId();

  const ask = async () => {
    if (!q.trim()) return;
    setBusy(true);
    try {
      const res = await askRag({ session_id, question: q, top_k: 5, doc_id: docId });
      const refs = Array.isArray(res?.sources) && res.sources.length
        ? '\n\nReferences: ' + res.sources.map(c => `[${c.doc_id} #${c.seq}]`).join(', ')
        : '';
      setA((res.answer || 'No answer.') + refs);
      setQ('');
    } catch (e) {
      setA('Request failed: ' + (e?.response?.data?.error || e.message));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <input value={q} onChange={e => setQ(e.target.value)} placeholder="Ask…" />
      <button disabled={busy} onClick={ask}>Ask</button>
      <pre>{a}</pre>
    </div>
  );
}
