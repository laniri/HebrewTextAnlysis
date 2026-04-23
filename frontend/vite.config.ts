import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Production builds use /coach/ base path for ALB path-based routing.
  // In dev mode, assets are served from root.
  base: process.env.NODE_ENV === 'production' ? '/coach/' : '/',
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/admin/config': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/admin/models': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
