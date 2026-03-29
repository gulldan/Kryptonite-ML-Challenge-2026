# Release Postmortem And Backlog v2

`KRYP-080` closes the release loop for the current Kryptonite stack. The main
conclusion is not that the model is "done"; it is that the release surface is
now reproducible enough to show its own gap honestly.

The current cycle succeeded at:

- turning QA, stress validation, benchmark freeze, bundle packaging, and
  rollback into reproducible repository workflows;
- making calibration, deploy notes, and operator checks explicit;
- producing a handoff surface that the next operator can validate without
  reconstructing release state from scratch.

The current cycle did not succeed at:

- promoting a learned ONNX or TensorRT backend to the live runtime path;
- turning teacher-heavy or advanced augmentation work into must-have release
  evidence;
- removing the mtime-sensitive threshold activation risk from rollout.

## What Worked

- Deploy and QA moved from scattered steps to a real release chain:
  `docs/end-to-end-regression-suite.md`, `docs/inference-stress-test.md`,
  `docs/final-benchmark-pack.md`, `docs/submission-release-bundle.md`, and
  `docs/release-runbook.md`.
- Calibration is now part of the release contract instead of a hidden local
  convention. The model card, runbook, and bundle docs all refer to the
  threshold artifact explicitly.
- Corruption-oriented robustness checks now show up as repeatable release
  evidence instead of notebook-only intuition.

## What Did Not Ship

- The runtime contract is ahead of the real learned export path. Multiple docs
  and tests call out the same boundary: the service now resolves
  `selected_backend` truthfully, but the active embedding implementation is
  still `feature_statistics`, so the learned export path is not live yet.
- Teacher PEFT, distillation, optional ReDimNet exploration, and
  consistency-loss work remained stretch items. They did not become part of the
  must-have release path.
- Advanced augmentation scope was not tied back to the final release candidate
  selection strongly enough to justify keeping it on the next critical path.

## Risks

- Threshold activation is still mtime-sensitive, so a newer stale calibration
  file can win until the release flow gets an explicit active-release pointer.
- Export benchmarking is easy to misread while ONNX/TensorRT are still planned
  surfaces rather than promoted serving paths.

## Backlog v2

The source of truth for the next-iteration backlog is
[`configs/release/release-postmortem-v2.toml`](../configs/release/release-postmortem-v2.toml).
That config feeds the reproducible postmortem builder and records the Linear
issue ids directly.

| Priority | Disposition | Linear | Item |
| --- | --- | --- | --- |
| `P0` | next iteration | `KVA-536` | Freeze the next export target family before re-running deployment work |
| `P0` | next iteration | `KVA-538` | Implement PyTorch -> ONNX export for the chosen release candidate |
| `P0` | next iteration | `KVA-539` | Build ONNX Runtime parity suite against the current PyTorch path |
| `P1` | next iteration | `KVA-544` | Package the real inference fallback chain after parity is proven |
| `P1` | next iteration | `KVA-540` | Build a TensorRT FP16 engine for the promoted ONNX candidate |
| `P1` | next iteration | `KVA-541` | Tune dynamic shape profiles around short/mid/long traffic |
| `P2` | next iteration | `KVA-542` | Benchmark PyTorch vs ORT vs TRT only after all three paths are real |
| `P2` | monitor | none yet | Replace mtime-based threshold discovery with an explicit active release pointer |
| `P3` | de-scoped | `KVA-531` | Keep teacher PEFT exploration out of the next must-have release plan |
| `P3` | de-scoped | `KVA-533` | De-scope distillation until one exportable student is locked |
| `P3` | de-scoped | `KVA-534` | De-scope consistency loss experiments from the next delivery milestone |
| `P3` | de-scoped | `KVA-535` | De-scope optional ReDimNet branch comparison until baseline export is stable |
| `P3` | de-scoped | `KVA-543` | De-scope INT8 feasibility until FP16 export is production-like |

## Reproducible Artifact

The checked-in config above can be rendered into a machine-readable report:

```bash
uv run python scripts/build_release_postmortem.py \
  --config configs/release/release-postmortem-v2.toml
```

The command writes:

```text
artifacts/release-postmortems/kryptonite-2026-release-v2/
├── release_postmortem.json
└── release_postmortem.md
```

That report fingerprints its evidence paths and keeps the backlog v2 in one
structured location, which makes future postmortem refreshes auditable instead
of ad hoc.
