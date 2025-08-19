import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/chat': 'http://localhost:8000',
      '/agent': 'http://localhost:8000',
      '/upload': 'http://localhost:8000',
      '/pdf': 'http://localhost:8000',
      '/healthz': 'http://localhost:8000',
      '/detect-persona-and-agents': 'http://localhost:8000',
      '/generate-chat-title': 'http://localhost:8000',
    }
  },
  build: {
    outDir: 'dist',
  },
})