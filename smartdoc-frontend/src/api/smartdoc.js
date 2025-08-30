// central API helpers for SmartDocQ
import axios from "axios";

export const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://localhost:5001/api/v1";

export async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await axios.post(`${API_BASE_URL}/upload`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data; // {message, doc_id, meta, text}
}

export async function embedDoc(doc_id) {
  const { data } = await axios.post(`${API_BASE_URL}/embed_doc`, { doc_id });
  return data; // {message, doc_id, chunks}
}

export async function askRag({ question, doc_id, session_id, top_k = 5, temperature, model }) {
  const payload = { question, doc_id, session_id, top_k };
  if (typeof temperature === "number") payload.temperature = temperature;
  if (model) payload.model = model;

  const { data } = await axios.post(`${API_BASE_URL}/ask`, payload);
  return data; // {answer, citations?, sources?, doc_id, session_id}
}

export async function listDocs() {
  const { data } = await axios.get(`${API_BASE_URL}/docs`);
  return data;
}

export async function healthz() {
  const { data } = await axios.get(`${API_BASE_URL}/healthz`);
  return data;
}
