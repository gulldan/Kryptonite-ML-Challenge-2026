"""Helpers for serving the built demo frontend bundle."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def render_demo_page() -> str:
    index_path = _frontend_dist_dir() / "index.html"
    if index_path.is_file():
        return index_path.read_text(encoding="utf-8")

    return dedent(
        """\
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>Kryptonite Demo Frontend Missing</title>
          <style>
            body {
              margin: 0;
              min-height: 100vh;
              display: grid;
              place-items: center;
              font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
              background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 30%),
                linear-gradient(180deg, #f8f5ef 0%, #efede6 100%);
              color: #172026;
            }

            main {
              width: min(720px, calc(100vw - 32px));
              padding: 28px;
              border-radius: 28px;
              border: 1px solid rgba(23, 32, 38, 0.12);
              background: rgba(255, 255, 255, 0.9);
              box-shadow: 0 24px 80px rgba(35, 44, 58, 0.12);
            }

            h1 {
              margin: 0 0 10px;
              font-family: "IBM Plex Serif", Georgia, serif;
              font-size: clamp(2rem, 4vw, 3.4rem);
              line-height: 0.95;
            }

            p {
              margin: 0 0 12px;
              color: #53606d;
              line-height: 1.65;
            }

            code {
              padding: 2px 6px;
              border-radius: 8px;
              background: rgba(15, 118, 110, 0.1);
            }
          </style>
        </head>
        <body>
          <main>
            <h1>Frontend bundle not found.</h1>
            <p>
              No prebuilt demo frontend was found under <code>apps/web/dist</code>.
            </p>
            <p>
              The JSON demo endpoints under <code>/demo/api/*</code> are already available.
            </p>
            <p>
              Add a separate web bundle only when this repository grows a dedicated frontend app.
            </p>
          </main>
        </body>
        </html>
        """
    )


def resolve_demo_frontend_assets_dir() -> Path | None:
    assets_dir = _frontend_dist_dir() / "assets"
    return assets_dir if assets_dir.is_dir() else None


def _frontend_dist_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "apps" / "web" / "dist"


__all__ = ["render_demo_page", "resolve_demo_frontend_assets_dir"]
