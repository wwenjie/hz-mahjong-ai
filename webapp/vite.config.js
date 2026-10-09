import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 前端构建产物输出到 webapp/dist，由 webapp/server.py 直接托管。
// 开发模式（npm run dev）下，/api 与 /api/stream 代理到 Python 后端。
export default defineConfig({
  plugins: [vue()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8848',
        changeOrigin: true,
      },
    },
  },
})
