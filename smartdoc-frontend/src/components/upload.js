import React, { useState } from "react";

function Upload({ onUpload }) {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:5001/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setMessage(data.message);
        if (onUpload) onUpload(data);
      } else {
        setMessage(data.error || "Upload failed");
      }
    } catch (err) {
      setMessage("Upload error: " + err.message);
    }
  };

  return (
    <div className="upload-container">
      <h2>Upload Document</h2>
      <input
        type="file"
        accept=".pdf,.txt,.docx,.csv,.xlsx,.jpg,.jpeg,.png"
        onChange={handleFileChange}
      />
      <button onClick={handleUpload}>Upload</button>
      <p>{message}</p>
    </div>
  );
}

export default Upload;
