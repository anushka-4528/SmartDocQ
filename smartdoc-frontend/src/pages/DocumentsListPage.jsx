import React from "react";
import { useNavigate } from "react-router-dom";

export default function DocumentsListPage() {
  const navigate = useNavigate();
  const docs = JSON.parse(localStorage.getItem("uploadedDocs") || "[]");

  const handleDelete = (id) => {
    const filtered = docs.filter((d) => d.id !== id);
    localStorage.setItem("uploadedDocs", JSON.stringify(filtered));
    window.location.reload();
  };

  return (
    <div className="documents-container">
      <h1 className="documents-title">📂 Uploaded Documents</h1>
      {!docs.length ? (
        <p className="no-docs">No documents uploaded yet.</p>
      ) : (
        <div className="documents-grid">
          {docs.map((doc) => (
            <div key={doc.id} className="document-card">
              <div>
                <p className="document-name">{doc.name}</p>
                <p className="document-meta">
                  {new Date(doc.date).toLocaleString()} • {doc.type || "unknown"}
                </p>
                {doc.text && <p className="document-preview">{doc.text.slice(0, 100)}...</p>}
              </div>
              <div className="document-actions">
                <button
                  className="view-btn"
                  onClick={() => navigate(`/qa/${doc.id}`, { state: { extractedText: doc.text } })}
                >
                  View
                </button>
                <button className="delete-btn" onClick={() => handleDelete(doc.id)}>Delete</button>
              </div>
            </div>
          ))}
        </div>
      )}
      <button className="back-btn" onClick={() => navigate("/upload")}>⬅ Back to Upload</button>
    </div>
  );
}
