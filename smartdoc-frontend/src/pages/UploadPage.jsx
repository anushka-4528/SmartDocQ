// src/pages/UploadPage.jsx
import React, { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FiUpload, FiChevronDown, FiFile, FiFileText, FiFilePlus } from "react-icons/fi";
import { uploadFile, embedDoc } from "../api/smartdoc";

// ✅ PDF.js (legacy build works well with CRA)
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf";
import pdfjsWorker from "pdfjs-dist/legacy/build/pdf.worker.min.js";

// Point PDF.js to the locally bundled worker (no CDN)
pdfjsLib.GlobalWorkerOptions.workerSrc = pdfjsWorker;

/** -------- Text extraction (client-side) -------- */
async function extractClientText(file) {
  const ext = file.name.split(".").pop().toLowerCase();

  if (ext === "txt") {
    return await file.text();
  }

  if (ext === "docx") {
    const mammoth = await import("mammoth");
    const buf = await file.arrayBuffer();
    const { value } = await mammoth.convertToPlainText({ arrayBuffer: buf });
    return value || "";
  }

  if (ext === "pdf") {
    const buf = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
    let text = "";
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      text += content.items.map((it) => it.str).join(" ") + "\n";
    }
    return text;
  }

  throw new Error("Unsupported file type. Please upload .txt, .docx, or .pdf");
}
/** ---------------------------------------------- */

export default function UploadPage() {
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);

  const pdfRef = useRef(null);
  const docxRef = useRef(null);
  const txtRef = useRef(null);

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setBusy(true);
      setStatus("Extracting text (client) …");
      const extractedText = await extractClientText(file);

      setStatus("Uploading to server …");
      const up = await uploadFile(file); // { doc_id, meta:{filename,pages,size_bytes}, text }
      if (!up?.doc_id) throw new Error("Backend did not return doc_id");

      const { doc_id, meta } = up;

      setStatus("Embedding & indexing …");
      await embedDoc(doc_id);

      // Persist preview for Documents page
      const docs = JSON.parse(localStorage.getItem("uploadedDocs") || "[]");
      docs.push({
        id: doc_id,
        name: meta?.filename || file.name,
        type: file.type,
        date: new Date().toISOString(),
        text: extractedText,
      });
      localStorage.setItem("uploadedDocs", JSON.stringify(docs));

      setStatus("Done! Redirecting …");
      navigate(`/qa/${doc_id}`, { state: { extractedText } });
    } catch (err) {
      console.error(err);
      alert(err?.message || "Upload failed");
    } finally {
      setBusy(false);
      e.target.value = ""; // reset input so same file can be re-selected
    }
  };

  return (
    <div className="hero">
      <h1>SmartDocQ with GenAI</h1>
      <p>Upload your document. We’ll extract, embed and index it, then you can chat with citations.</p>

      <div className="button-row">
        <div className="upload-dropdown">
          <button className="upload-btn" disabled={busy} onClick={() => setMenuOpen(!menuOpen)}>
            <FiUpload size={18} /> Upload Document <FiChevronDown size={16} />
          </button>

          <div className={`upload-menu ${menuOpen ? "show" : ""}`}>
            <div className="upload-item" onClick={() => pdfRef.current?.click()}>
              <FiFile size={16} /> Upload PDF
            </div>
            <div className="upload-item" onClick={() => docxRef.current?.click()}>
              <FiFileText size={16} /> Upload Word
            </div>
            <div className="upload-item" onClick={() => txtRef.current?.click()}>
              <FiFilePlus size={16} /> Upload Text
            </div>
          </div>
        </div>

        <button className="hero-button" onClick={() => navigate("/documents")}>
          📂 View All Documents
        </button>
      </div>

      {/* Hidden inputs */}
      <input type="file" ref={pdfRef} accept=".pdf" onChange={handleUpload} style={{ display: "none" }} />
      <input type="file" ref={docxRef} accept=".docx" onChange={handleUpload} style={{ display: "none" }} />
      <input type="file" ref={txtRef} accept=".txt" onChange={handleUpload} style={{ display: "none" }} />

      {status && <p style={{ marginTop: 20 }}>{status}</p>}
    </div>
  );
}
