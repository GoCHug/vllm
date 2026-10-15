import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { viteSingleFile } from 'vite-plugin-singlefile'

// base './' + singlefile: 产物仅一个 dist/index.html, FastAPI serve 与 file:// 双击(配 pack)两用
export default defineConfig({
  base: './',
  plugins: [vue(), viteSingleFile()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://localhost:8000', changeOrigin: true } }
  }
})
