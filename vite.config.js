import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev-only: serve the static reveal.js deck at /slides and /slides/.
// Without this, Vite's SPA fallback returns the React app for the bare
// directory path, which both shows the wrong page and makes the relative
// "./slides/" link recurse (/slides/slides/...). In production GitHub Pages
// serves public/slides/index.html for the directory automatically, so this
// plugin only matters for `vite dev`/`vite preview`.
function serveSlidesDeck() {
  const slidesIndex = fileURLToPath(new URL('./public/slides/index.html', import.meta.url))
  const middleware = (req, res, next) => {
    const path = req.url.split('?')[0]
    if (path === '/slides') {
      res.statusCode = 301
      res.setHeader('Location', '/slides/')
      return res.end()
    }
    if (path === '/slides/') {
      res.setHeader('Content-Type', 'text/html; charset=utf-8')
      return res.end(readFileSync(slidesIndex))
    }
    next()
  }
  return {
    name: 'serve-slides-deck',
    configureServer(server) {
      server.middlewares.use(middleware)
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware)
    },
  }
}

export default defineConfig({
  plugins: [react(), serveSlidesDeck()],
  base: "./",
});
