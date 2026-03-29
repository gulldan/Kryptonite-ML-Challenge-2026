# Kryptonite Speaker Verification Model Card

## Status And Release Scope

This repository contains the training, evaluation and deployment contracts for the Kryptonite speaker-verification stack.
The current checked-in runtime is intentionally honest: it validates the runtime surface, artifacts and telemetry,
but the active raw-audio implementation is still `feature_statistics` unless a real learned bundle is explicitly promoted.

## System Summary

Canonical flow:

1. raw audio input
2. explicit `16 kHz` mono preprocessing
3. optional VAD / trimming
4. `80`-bin Fbank frontend
5. chunking and pooling
6. embedding extraction
7. cosine scoring
8. threshold-based decision

See [system-architecture-v1.md](./system-architecture-v1.md) for the architectural map and [release-runbook.md](./release-runbook.md) for the operational runtime view.

## Intended Use

Appropriate use:

- internal speaker-verification model development
- candidate comparison and benchmark freeze
- runtime/demo validation
- controlled local or GPU-server demo flows

Not appropriate use:

- biometric authentication claims without candidate-specific benchmark and threshold freeze
- forensic/legal identity claims
- fairness claims across demographic slices not evaluated in-repo
- production claims based only on the checked-in demo bundle

## Training Data Classes

| Data class | Status | Current role | Notes |
| --- | --- | --- | --- |
| Dataton Kryptonite 2026 raw challenge data | `blocked` | target dataset | not yet treated as materialized in-repo source of truth |
| FFSVC 2022 | `conditional` | engineering surrogate | useful for pipeline bring-up, not a final challenge claim |
| MUSAN / RIR / synthetic channel transforms | `approved` | robustness and augmentation | open or reproducible support sources |
| Synthetic demo subset | smoke only | local runtime validation | not evidence of final model quality |

See [data.md](./data.md) for the short operational summary and [archive/dataset-inventory.md](./archive/dataset-inventory.md) for the detailed inventory.

## Bias And Robustness Caveats

The repository can currently make only engineering-level robustness claims.
Known caveats:

- no demographic fairness evaluation is checked in
- surrogate data is not identical to the target challenge data
- robustness is partly simulated through noise, RIR, distance, codec and silence transforms
- demo success should not be read as broad deployment robustness

## Limitations And Risks

- the active runtime validates contracts and observability more than final learned-model quality
- thresholds are candidate-specific and must be regenerated for real releases
- enrollment caches are valid only for the matching model bundle
- archive material contains historical decisions and experiments; it should not override the canonical top-level docs

## Deployment Notes

Operational source of truth:

- runtime startup and troubleshooting: [release-runbook.md](./release-runbook.md)
- config map: [configuration.md](./configuration.md)
- detailed export/deployment notes: [archive/onnx-export.md](./archive/onnx-export.md), [archive/triton-deployment.md](./archive/triton-deployment.md), [archive/tensorrt-fp16-engine.md](./archive/tensorrt-fp16-engine.md)

## Release Checklist

1. Freeze training and deploy configs.
2. Freeze offline quality and threshold artifacts.
3. Validate runtime smoke against the intended artifact set.
4. Confirm cache compatibility with the active model bundle.
5. Package the final handoff only after candidate-specific evidence is real.
