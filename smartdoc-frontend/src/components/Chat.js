import React, { useState } from 'react'
import { askRag } from '../api/smartdoc'
import { getSessionId } from '../utils/session'

export default function Chat({ doc_uuid }){
  const [q,setQ] = useState('')
  const [a,setA] = useState('')
  const [busy,setBusy] = useState(false)
  const session_id = getSessionId()

  const ask = async () => {
    if(!q.trim()) return
    setBusy(true)
    const res = await askRag({ session_id, query: q, k: 5, doc_uuid })
    const refs = res?.citations?.length ? '\n\nReferences: ' + res.citations.map(c => `[${c.doc} #${c.seq}]`).join(', ') : ''
    setA((res.answer || 'No answer.') + refs)
    setQ('')
    setBusy(false)
  }

  return (
    <div>
      <input value={q} onChange={e=>setQ(e.target.value)} placeholder="Ask…" />
      <button disabled={busy} onClick={ask}>Ask</button>
      <pre>{a}</pre>
    </div>
  )
}
