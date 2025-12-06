import React from 'react';
import { Routes, Route, NavLink, Link, useNavigate } from 'react-router-dom';
import ChatPage from './ChatPage';
import UploadPage from './UploadPage';
import FileDetailsPage from './FileDetailsPage';
import TranscriptPage from './TranscriptPage';
import MinutesPage from './MinutesPage';
import ProcessedBatchDetailsPage from './ProcessedBatchDetailsPage';
import LoginPage from './LoginPage';
import RegisterPage from './RegisterPage';
import TranscriptsPage from './TranscriptsPage';
import ApiKeysPage from './ApiKeysPage';
import { useAuth } from './AuthContext';
import './App.css';
import { useApiTokens } from './ApiTokensContext';

function App() {
  const { isAuthenticated, logout } = useAuth();
  const { clearTokens } = useApiTokens();
  const navigate = useNavigate();

  const handleLogout = () => {
    clearTokens();
    logout();
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      <nav className="bg-gray-900 p-4 shadow-md">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between text-white">
          <ul className="flex space-x-4">
          <li>
            <NavLink
              to="/"
              className={({ isActive }) =>
                `px-3 py-2 rounded-md text-sm transition ${
                  isActive
                    ? 'bg-white text-gray-900 font-semibold shadow-sm border border-gray-300'
                    : 'text-white hover:bg-gray-800/40 hover:text-white border border-transparent'
                }`
              }
              style={({ isActive }) => (isActive ? { color: '#111827' } : undefined)}
            >
              Upload i Przetwarzanie
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/chat"
              className={({ isActive }) =>
                `px-3 py-2 rounded-md text-sm transition ${
                  isActive
                    ? 'bg-white text-gray-900 font-semibold shadow-sm border border-gray-300'
                    : 'text-white hover:bg-gray-800/40 hover:text-white border border-transparent'
                }`
              }
              style={({ isActive }) => (isActive ? { color: '#111827' } : undefined)}
            >
              Chat
            </NavLink>
          </li>
          {isAuthenticated && (
            <li>
              <NavLink
                to="/transcripts"
                className={({ isActive }) =>
                  `px-3 py-2 rounded-md text-sm transition ${
                    isActive
                    ? 'bg-white text-gray-900 font-semibold shadow-sm'
                    : 'text-white hover:bg-gray-800/40 hover:text-white'
                  }`
                }
              style={({ isActive }) => (isActive ? { color: '#111827' } : undefined)}
              >
                Transkrypcje
              </NavLink>
            </li>
          )}
          {isAuthenticated && (
            <li>
              <NavLink
                to="/api-keys"
                className={({ isActive }) =>
                  `px-3 py-2 rounded-md text-sm transition ${
                    isActive
                    ? 'bg-white text-gray-900 font-semibold shadow-sm border border-gray-300'
                    : 'text-white hover:bg-gray-800/40 hover:text-white border border-transparent'
                  }`
                }
              style={({ isActive }) => (isActive ? { color: '#111827' } : undefined)}
              >
                Klucze API
              </NavLink>
            </li>
          )}
          </ul>
          <div className="flex items-center space-x-3">
            {isAuthenticated ? (
              <button
                onClick={handleLogout}
                className="rounded-md px-3 py-1 text-sm font-semibold text-white transition hover:bg-white/10"
              >
                Wyloguj
              </button>
            ) : (
              <>
                <Link
                  to="/login"
                  className="rounded-md px-3 py-1 text-sm font-semibold text-white transition hover:bg-white/10"
                >
                  Zaloguj
                </Link>
                <Link
                  to="/register"
                  className="rounded-md px-3 py-1 text-sm font-semibold text-white transition hover:bg-white/10"
                >
                  Rejestracja
                </Link>
              </>
            )}
          </div>
        </div>
      </nav>

      <main className="flex-grow max-w-7xl mx-auto w-full">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/processed_files/:fileId/details" element={<FileDetailsPage />} />
          <Route path="/processed_files/:fileId/transcript" element={<TranscriptPage />} />
          <Route path="/processed_files/:fileId/minutes" element={<MinutesPage />} />
          <Route path="/processed_batches/:batchJobId/details" element={<ProcessedBatchDetailsPage />} />
          <Route path="/transcripts" element={<TranscriptsPage />} />
          <Route path="/api-keys" element={<ApiKeysPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          {/* Możesz dodać trasę 404 lub przekierowanie do strony głównej dla nieznanych ścieżek */}
        </Routes>
      </main>
    </div>
  );
}

export default App;
