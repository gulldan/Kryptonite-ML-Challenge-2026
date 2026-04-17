# AGENTS.md

This directory is an archived organizer baseline kept for reference only.

## Status

- `baseline/` is not the repository source of truth for tooling, packaging, or entrypoints
- the canonical project toolchain is defined in the repository-root `AGENTS.md`
- `baseline/requirements.txt` and `python -m venv` instructions are preserved only to document the original organizer baseline workflow

## Editing rules

- Do not import new production logic from `baseline/` into `src/kryptonite/`
- Do not point README, CI, Docker, or deployment entrypoints at `baseline/` unless the user explicitly asks for organizer-baseline compatibility work
- If you update `baseline/README.md`, keep the archival status explicit
