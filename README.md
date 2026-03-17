# site-template

A minimalist project website template.

The only file to edit is [`src/content.jsx`](src/content.jsx).

Demo: https://shunchizhang.github.io/site-template/

## Setup

```sh
git clone https://github.com/ShunchiZhang/site-template.git your-site-name
cd your-site-name
npm install
npm run dev
```

Preview: http://localhost:5173/ (Automatically updated when you save [`src/content.jsx`](src/content.jsx).)

## Deploy to GitHub Pages

1. Specify your site name in [`vite.config.js`](vite.config.js):
   ```js
   base: '/your-site-name/',
   ```
2. Run `npm run deploy`

Live at `https://<username>.github.io/<your-site-name>/`.
