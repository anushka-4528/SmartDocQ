import api from './client';

// POST /upload  -> returns { message, doc_id, meta, text }
export async function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);
  const { data } = await api.post('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

// POST /embed_doc  body: { doc_id } -> { message, doc_id, chunks }
export async function embedDoc(doc_id, chunk_size = 1200, overlap = 200) {
  const { data } = await api.post('/embed_doc', { doc_id, chunk_size, overlap });
  return data;
}

// POST /ask  body: { question, doc_id, session_id, top_k, ... } -> parsed + {sources?}
export async function askRag({ question, doc_id, session_id, top_k = 5, style = 'concise', citation_mode = true, model, temperature }) {
  const payload = { question, doc_id, session_id, top_k, style, citation_mode };
  if (model) payload.model = model;
  if (temperature != null) payload.temperature = temperature;

  const { data } = await api.post('/ask', payload);
  return data; // { answer, sources:[{id,score,doc_id,seq,snippet}], doc_id, session_id, ... }
}
