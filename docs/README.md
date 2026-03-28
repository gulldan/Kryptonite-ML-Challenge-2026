# Docs

Curated architecture notes, runbooks, model cards, and benchmark summaries belong here.

Exploratory notes should stay in `notebooks/` until they become stable enough to promote into documentation.

Key serving notes:

- `docs/epic-00-research-baseline.md` for the single EPIC-00 closeout entrypoint that ties together rules, task contract, verification protocol, architecture, and experiment ordering
- `docs/model-task-contract.md` for the repository-level ADR that fixes the primary verification task, compatible identification modes, trial semantics, and handoff artifacts
- `docs/internal-verification-protocol.md` for the clean plus corrupted verification-protocol snapshot, required slice fields, and the strict completeness gate for full rebuilds
- `docs/experiment-matrix-v1.md` for the sequenced experiment plan, GPU-hour budget ranges, and the explicit must-have vs stretch split
- `docs/teacher-peft.md` for the runnable WavLM / w2v-BERT stretch-teacher branch, PEFT limits, and checkpoint layout
- `docs/campp-distillation.md` for the runnable CAM++ student-distillation recipe, teacher-guided losses, and baseline-vs-distilled comparison contract
- `docs/campp-consistency.md` for the runnable CAM++ clean/corrupted consistency recipe, pairwise invariance losses, and built-in robust-dev ablation
- `docs/system-architecture-v1.md` for the end-to-end pipeline diagram, module ownership, interfaces, logging points, and export/serve placement
- `docs/onnx-export.md` for the real CAM++ encoder-only ONNX bundle, validation scope, and handoff limits before parity promotion
- `docs/tensorrt-fp16-engine.md` for the repo-native FP16 engine build workflow, validation gates, and metadata promotion rules
- `docs/web-demo.md` for the browser demo and runtime entrypoints
- `docs/triton-deployment.md` for the encoder-boundary Triton model repository, launch flow, and TensorRT handoff constraints
- `docs/inference-observability.md` for structured logs and Prometheus-compatible metrics
- `docs/end-to-end-regression-suite.md` for the release-oriented serving regression contract
- `docs/inference-stress-test.md` for the serving-path stress matrix, malformed-input contract, and hard-limit reporting flow
- `docs/final-benchmark-pack.md` for the release pack that freezes quality/latency/memory and exact configs
- `docs/final-family-decision.md` for the frozen export-target family ADR that feeds the next ONNX/parity cycle
- `docs/model-card.md` for the release scope, training-data classes, caveats, and deployment assumptions
- `docs/release-runbook.md` for preflight, rollout, monitoring, and rollback procedures
- `docs/submission-release-bundle.md` for the handoff bundle that packages configs, model artifacts, docs, demo assets, and an optional archive
- `docs/release-postmortem.md` for the release retrospective, backlog v2, and the reproducible postmortem workflow
- `docs/dataton-rules-matrix.md` for the competition-facing allow/deny/unknown matrix and risk register around external data, models, and augmentations
