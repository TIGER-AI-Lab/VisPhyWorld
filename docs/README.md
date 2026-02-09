# VisPhyWorld Project Page

This folder contains a static project page based on the Academic Project Page Template:
`https://github.com/eliahuhorwitz/Academic-project-page-template`.

Main entry: `docs/index.html`.

## Local preview

Open `docs/index.html` directly in a browser, or run a simple server:

```bash
python -m http.server 8000 --directory docs
```

Then visit `http://localhost:8000`.

## GitHub Pages

In your GitHub repo settings:

1. **Settings → Pages**
2. **Build and deployment**: Deploy from a branch
3. **Branch**: `main` (or your default branch)
4. **Folder**: `/docs`

GitHub will publish the site at `https://<username>.github.io/<repo>/`.

## Customize

- Update the placeholder URLs in `docs/index.html` (e.g., GitHub username, paper link).
- Replace teaser/sample videos in `docs/static/videos/`.
