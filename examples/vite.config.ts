import { defineConfig } from 'vite';
import { fileURLToPath, URL } from 'node:url';

const TFLITE_WASM_CDN =
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm';

export default defineConfig({
  resolve: {
    alias: {
      uww: fileURLToPath(new URL('../src/index.ts', import.meta.url)),
      // tfjs-tflite ships a broken `module` entry that imports a non-existent
      // file. The FESM bundle is the only usable ES module in the package.
      '@tensorflow/tfjs-tflite': fileURLToPath(
        new URL(
          '../node_modules/@tensorflow/tfjs-tflite/dist/tf-tflite.fesm.js',
          import.meta.url
        )
      ),
    },
  },
  plugins: [
    {
      // tfjs-tflite runs an unconditional `loader.load(true)` at module
      // import time using a hard-coded empty wasmPath. That probes the
      // page origin for `tflite_web_api_cc*.js` and 404s. Redirect those
      // probes to the CDN so the dev console stays clean.
      name: 'uww:tflite-wasm-redirect',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url && /^\/tflite_web_api_cc[^/]*\.(?:js|wasm)$/.test(req.url)) {
            res.writeHead(302, { Location: `${TFLITE_WASM_CDN}${req.url}` });
            res.end();
            return;
          }
          next();
        });
      },
    },
  ],
  server: {
    port: 5173,
    open: true,
  },
});
