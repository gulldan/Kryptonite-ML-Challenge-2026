# AGENTS.md

Local rules for `src/kryptonite/eda/`.

- Keep EDA reproducible and rerunnable from code.
- Write reusable profiling, leakage-check, and auditing helpers here.
- Save generated outputs to `artifacts/eda/` unless they are intentionally curated into `docs/`.
- Do not hide training code or ingestion side effects inside EDA modules.
- Keep each EDA source file below `600` lines; split large workflows into focused modules or a package before extending them further.
