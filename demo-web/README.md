# Retention — web demo

A walkthrough of the Retention pipeline: upload an educational video, score it
with TRIBE v2, run the generator/evaluator loop, and compare brain analytics for
the original vs improved video.

## Run
```bash
cd demo-web
npm install
npm run dev      # open the printed localhost URL
```

## Controls
- **Next** / **→** — advance a page
- **dots** — jump to a page

## Structure
```
src/
  App.jsx                page router + chrome
  hooks/usePages.js      page navigation
  data/
    pages.js             page list
    demoData.js          content (assets, scores, copy)
  components/            one per page / widget
public/assets/           media
```

## Assets
The background video is gitignored — add `demo-web/public/assets/background.mp4`.
