    // frontend/vite.config.js
    import { defineConfig } from 'vite'
    import react from '@vitejs/plugin-react-swc' // lub '@vitejs/plugin-react' jeśli nie używasz SWC
    import tailwindcss from '@tailwindcss/vite'

    // https://vitejs.dev/config/
    export default defineConfig({
      plugins: [react(),tailwindcss()],
      server: {
        proxy: {
          '/token': {
            target: 'http://localhost:7777', // api_app.py for token endpoint
            changeOrigin: true,
            // Endpoint /token nie wymaga autoryzacji do uzyskania tokena
          },
          '/register': {
            target: 'http://localhost:7777', // api_app.py for register endpoint
            changeOrigin: true,
          },
          '/chat/query': {
            target: 'http://localhost:8000', // FastAPI Minimal MCP server for chat
            changeOrigin: true,
            rewrite: (path) => path.replace(/^\/chat\/query/, '/chat/query'),
          },
          // Proxy for all other API endpoints to api_app.py
          '/prompts': {
            target: 'http://localhost:7777', // api_app.py for prompts
            changeOrigin: true,
          },
          '/upload_multiple': {
            target: 'http://localhost:7777', // api_app.py for file upload
            changeOrigin: true,
          },
          '/get_enrolled_speakers': {
            target: 'http://localhost:7777', // api_app.py for speakers
            changeOrigin: true,
          },
          '/enroll_speaker_direct': {
            target: 'http://localhost:7777', // api_app.py for speaker enrollment
            changeOrigin: true,
          },
          '/delete_speaker': {
            target: 'http://localhost:7777', // api_app.py for speaker deletion
            changeOrigin: true,
          },
          '/user/transcripts': {
            target: 'http://localhost:7777', // api_app.py for user transcripts CRUD
            changeOrigin: true,
          },
          '/process_batch_status/stream': {
            target: 'http://localhost:7777', // api_app.py for batch status stream
            changeOrigin: true,
            rewrite: (path) => path, // Zachowaj pełną ścieżkę dla backendu
          },
          '/processed_batches': {
            target: 'http://localhost:7777', // api_app.py for processed batch details
            changeOrigin: true,
            rewrite: (path) => path, // Zachowaj pełną ścieżkę dla backendu
            bypass: (req) => {
              if (req.headers.accept && req.headers.accept.includes('text/html')) {
                console.log('Vite proxy bypass for processed_batches path');
                return req.url;
              }
            },
          },
          '/processed_files': {
            target: 'http://localhost:7777', // api_app.py for processed batch details
            changeOrigin: true,
            rewrite: (path) => path,
            bypass: (req) => {
              if (req.headers.accept && req.headers.accept.includes('text/html')) {
                console.log('Vite proxy bypass for processed_files path');
                return req.url;
              }
            },
          },
          '/health': {
            target: 'http://localhost:7777', // api_app.py for processed batch details
            changeOrigin: true,
          },
          // Catch-all for any other API calls to api_app.py
          // This is a more general proxy that should handle other endpoints for the upload page
          // that are not explicitly listed above (e.g., /process_file/ which is also on api_app.py)
          // Make sure this is the LAST proxy rule, so more specific rules are applied first.
          // Temporary catch-all to ensure all API requests are proxied
          // '/': {
          //   target: 'http://localhost:7777', // api_app.py
          //   changeOrigin: true,
          //   headers: {
          //     'Authorization': 'Bearer ' + (typeof window !== 'undefined' ? localStorage.getItem('authToken') : '')
          //   }
          // },
        },
        fs: {
          cachedChecks: true // Może przyspieszyć działanie serwera deweloperskiego
        }
      }
    })