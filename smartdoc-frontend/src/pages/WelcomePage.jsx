import React from "react";
import { useNavigate } from "react-router-dom";
import { FiLogIn, FiUserPlus } from "react-icons/fi";

export default function WelcomePage() {
  const navigate = useNavigate();
  return (
    <div className="hero">
      <h1>SmartDocQ</h1>
      <p>Upload, index and ask questions with grounded answers and citations.</p>
      <div className="button-row">
        <button className="hero-button" onClick={() => navigate("/upload")}>
          <FiLogIn style={{ marginRight: 8 }} /> Get Started
        </button>
        <button className="hero-button" onClick={() => navigate("/documents")}>
          <FiUserPlus style={{ marginRight: 8 }} /> View Documents
        </button>
      </div>
    </div>
  );
}
