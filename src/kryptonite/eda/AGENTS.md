# AGENTS.md

Local rules for `src/kryptonite/eda/`.

- Keep reusable exploratory analysis, leakage checks, profiling, validation, and report helpers here.
- Do not put training orchestration, serving adapters, or notebook-only logic in this package.
- Prefer artifact-producing functions that can be rerun from `scripts/` over interactive-only analysis.
- Keep dashboard/UI code outside this package; dashboards should consume artifacts produced here.
