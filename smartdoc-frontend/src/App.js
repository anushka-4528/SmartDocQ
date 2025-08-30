import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';

import SplashScreen from './pages/SplashScreen.jsx';
import WelcomePage from './pages/WelcomePage.jsx';
import UploadPage from './pages/UploadPage.jsx';
import DocumentsListPage from './pages/DocumentsListPage.jsx';
import DocumentQAPage from './pages/DocumentQAPage.jsx';

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<SplashScreen />} />
      <Route path="/welcome" element={<WelcomePage />} />
      <Route path="/upload" element={<UploadPage />} />
      {/* allow /uploads to redirect to /upload */}
      <Route path="/uploads" element={<Navigate to="/upload" replace />} />
      <Route path="/documents" element={<DocumentsListPage />} />
      <Route path="/qa/:id" element={<DocumentQAPage />} />
      {/* catch-all to welcome (optional) */}
      {/* <Route path="*" element={<Navigate to="/welcome" replace />} /> */}
    </Routes>
  );
}
