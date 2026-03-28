# site-template

Fork, edit, live -- just customizing one readable file.

Demo: https://shunchi.dev/site-template/

## Quick Start

1. [Fork](https://github.com/ShunchiZhang/site-template/fork) this repo.
2. Rename repo to your site name:
   1. Settings (Top Bar)
   2. General (Left Sidebar)
   3. Repository name -> Input your site name -> Click "Rename"
3. Enable Github Pages:
   1. Settings (Top Bar)
   2. Pages (Left Sidebar)
   3. Source -> Change "Deploy from a branch" to "GitHub Actions"
4. Run deployment workflow:
   1. Actions (Top Bar)
   2. Deploy to GitHub Pages (Left Sidebar)
   3. Click "Run workflow"
5. Your site will be live at `https://<your-username>.github.io/<your-site-name>/` after a few minutes.
6. Edit [`src/content.jsx`](src/content.jsx) directly on GitHub, commit, and your site will be updated automatically.

## Local Development

Alternatively, you can customize and preview your site locally.

```sh
git clone https://github.com/your-username/your-site-name
cd your-site-name
npm install
npm run dev
```

- Edit [`src/content.jsx`](src/content.jsx) and live preview at http://localhost:5173/.
- Your site will be updated automatically if you push your commits to GitHub.
