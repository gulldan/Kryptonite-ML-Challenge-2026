# Docs

Curated architecture notes, runbooks, model cards, and benchmark summaries belong here.

Exploratory notes should stay in `notebooks/` until they become stable enough to promote into documentation.

Key serving notes:

- `docs/web-demo.md` for the browser demo and runtime entrypoints
- `docs/triton-deployment.md` for the encoder-boundary Triton model repository, launch flow, and TensorRT handoff constraints
- `docs/inference-observability.md` for structured logs and Prometheus-compatible metrics
- `docs/end-to-end-regression-suite.md` for the release-oriented serving regression contract
- `docs/inference-stress-test.md` for the serving-path stress matrix, malformed-input contract, and hard-limit reporting flow
- `docs/final-benchmark-pack.md` for the release pack that freezes quality/latency/memory and exact configs
- `docs/model-card.md` for the release scope, training-data classes, caveats, and deployment assumptions
- `docs/release-runbook.md` for preflight, rollout, monitoring, and rollback procedures
