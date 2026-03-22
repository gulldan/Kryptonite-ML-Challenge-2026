# API App

This directory is reserved for thin serving entrypoints and transport adapters.

Do not place business logic here. Import reusable behavior from `src/kryptonite/serve/` and adjacent core modules.

The initial HTTP entrypoint lives in `apps/api/main.py` and delegates runtime validation and request handling to `src/kryptonite/serve/`.
