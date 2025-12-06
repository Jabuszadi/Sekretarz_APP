import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { BrowserRouter } from 'react-router-dom';
import { AuthProvider } from './AuthContext';
import { ApiTokensProvider } from './ApiTokensContext.jsx';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <ApiTokensProvider>
        <AuthProvider>
          <App />
        </AuthProvider>
      </ApiTokensProvider>
    </BrowserRouter>
  </StrictMode>,
)
