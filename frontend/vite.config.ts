import { fileURLToPath } from 'node:url';
import { defineConfig, normalizePath } from 'vite';
import react from '@vitejs/plugin-react';
import { viteStaticCopy } from 'vite-plugin-static-copy';

const pdfAssets = ['cmaps', 'standard_fonts', 'iccs', 'wasm', 'LICENSE'];

export default defineConfig({
  plugins: [
    react(),
    viteStaticCopy({
      targets: pdfAssets.map(resource => ({
        src: normalizePath(fileURLToPath(new URL(`./node_modules/pdfjs-dist/${resource}`, import.meta.url))),
        dest: 'pdf-assets',
        rename: { stripBase: 2 },
      })),
    }),
  ],
  server: {
    port: 5173,
    strictPort: true,
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
  preview: { proxy: { '/api': 'http://127.0.0.1:8000' } },
});
