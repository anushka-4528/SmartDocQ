import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";

export default function SplashScreen() {
  const navigate = useNavigate();
  useEffect(() => {
    const t = setTimeout(() => navigate("/welcome"), 900);
    return () => clearTimeout(t);
  }, [navigate]);

  return (
    <div className="hero">
      <h1>SmartDocQ</h1>
      <p>Your Smart Document Assistant</p>
    </div>
  );
}
