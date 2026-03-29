# Kryptonite Speaker Verification Model Card

## Status And Release Scope

This repository contains the training, evaluation, and deployment contracts for
the Kryptonite 2026 speaker-verification stack.

Two important scope boundaries are intentional:

- the checked-in deployment/demo flow is real enough to validate runtime,
  artifacts, thresholds, telemetry, and operational behavior;
- the checked-in `artifacts/model-bundle/` path is still a smoke/demo contract
  unless it is explicitly replaced by a frozen release candidate bundle.

In other words, this card documents the release expectations for any candidate
built through this repository, while also being explicit that the current
checked-in demo bundle is not itself a production-grade speaker-recognition
model.

For the repository-level ADR that fixes the primary task formulation, compatible
identification modes, and expected handoff artifacts, see
[model-task-contract.md](./model-task-contract.md).

## System Summary

The repository is organized around one canonical verification flow:

1. raw audio input;
2. explicit `16 kHz` mono normalization;
3. optional VAD / trimming;
4. `80`-bin log-Mel Fbank extraction;
5. utterance chunking and pooling;
6. embedding extraction;
7. cosine scoring against enrolled speaker references;
8. threshold-based decision profiles for `balanced`, `min_dcf`, `demo`, and
   `production`.

Current training candidates are:

- CAM++ baseline and staged fine-tuning recipes;
- ERes2NetV2 baseline.

Current serving behavior is more conservative than the long-term roadmap:

- the runtime surface already exposes `selected_backend`, model-bundle metadata,
  artifact checks, thresholds, and telemetry;
- the actual raw-audio embedding implementation is still
  `feature_statistics`, not a deployed learned ONNX/TensorRT encoder.

That distinction matters operationally: a healthy service is proof that the
deploy contract is wired correctly, not proof that a final learned export path
is already live.

## Intended Use

Appropriate use:

- internal model development for far-field speaker verification;
- candidate comparison and benchmark freeze;
- release QA for scoring, thresholds, stress behavior, and deployment
  readiness;
- local or controlled demo flows through the FastAPI runtime.

Not appropriate use:

- biometric authentication without a frozen candidate-specific benchmark pack
  and threshold calibration;
- forensic or legal identity claims;
- fairness claims across demographic groups, accents, or languages;
- production deployment that depends on the checked-in synthetic demo bundle.

## Training Data Classes

The repository keeps data provenance explicit through
[`configs/data-inventory/allowed-sources.toml`](../configs/data-inventory/allowed-sources.toml)
and [`docs/dataset-inventory.md`](./dataset-inventory.md).

| Data class | Status | Current role | Notes |
| --- | --- | --- | --- |
| Dataton Kryptonite 2026 raw challenge data | `blocked` | target train/dev/eval | Real competition data is not yet materialized in-repo, so no release claim should pretend otherwise. |
| FFSVC 2022 dev/eval bundle | `conditional` | engineering surrogate | Useful for pipeline bring-up and verification tooling, but not a challenge-legal final training recipe. |
| FFSVC 2020 main and supplementary data | `conditional` | potential external training | Requires explicit access handling and overlap policy before use. |
| VoxCeleb 1/2 | `blocked` | not used | Official source path is not reproducible from this repo today. |
| MUSAN | `approved` | additive noise augmentation | Default open noise source for corruption work. |
| OpenSLR RIR and noise database | `approved` | reverb / room simulation | Approved base for reverberation and far-field simulation. |
| Synthetic codec and channel transforms | `approved` | channel / codec robustness | Prefer deterministic FFmpeg transforms over another corpus. |
| Synthetic demo subset | smoke only | local deployment validation | Only for runtime/bootstrap checks, not for model-quality claims. |

## Preprocessing And Feature Contract

The current repository-wide audio contract is intentionally explicit:

- normalization target: `16 kHz`, mono, `PCM16 WAV`;
- loudness normalization: supported, disabled by default in `configs/base.toml`;
- VAD modes: `none`, `light`, `aggressive`;
- Fbank frontend: `80` mel bins, `25 ms` window, `10 ms` hop, `512` FFT;
- training chunking: `1-4 s`;
- eval/demo chunking: `4 s` windows with `1 s` overlap by default.

The most important supporting notes are:

- [audio-normalization.md](./audio-normalization.md)
- [audio-vad-trimming.md](./audio-vad-trimming.md)
- [audio-fbank-extraction.md](./audio-fbank-extraction.md)
- [audio-chunking-policy.md](./audio-chunking-policy.md)

## Evaluation And Release Evidence

A candidate should not be treated as release-ready from training metrics alone.
The repository now expects a release decision to reference all of the following:

- offline quality:
  [evaluation-package.md](./evaluation-package.md)
- threshold freeze:
  [threshold-calibration.md](./threshold-calibration.md)
- serving regression:
  [end-to-end-regression-suite.md](./end-to-end-regression-suite.md)
- serving stress and hard limits:
  [inference-stress-test.md](./inference-stress-test.md)
- final multi-candidate comparison and provenance freeze:
  [final-benchmark-pack.md](./final-benchmark-pack.md)

The benchmark pack is the intended handoff artifact for final release review,
because it stages exact configs, quality reports, threshold bundles, stress
results, and memory observations in one place.

## Bias And Robustness Caveats

The current repository can only make narrow, engineering-oriented robustness
claims.

Known caveats:

- no demographic metadata is tracked in the repository datasets, so no fairness
  evaluation across age, gender, accent, dialect, or language is available;
- the current surrogate is heavily skewed toward very low loudness and
  high-silence recordings, which is useful for robustness work but is not
  representative of every deployment environment;
- corruption robustness is partly simulated through additive noise, RIR,
  distance, codec, and silence transforms, which means real device/channel
  shifts can still differ materially from the lab setup;
- the synthetic demo subset is intentionally clean and tiny, so demo success is
  not evidence of broad deployment robustness.

Supporting data-quality notes:

- [ffsvc2022-surrogate.md](./ffsvc2022-surrogate.md)
- [audio-corrupted-dev-suites.md](./audio-corrupted-dev-suites.md)

## Limitations And Risks

Current release limitations that should be called out during review:

- the checked-in demo/runtime path validates deployment shape more than final
  learned-model quality;
- `requested_backend` reflects deployment intent, while `selected_backend`
  reflects the resolved runtime path after fallback selection;
- the active inferencer implementation still needs to be checked separately
  through `inferencer.implementation`;
- threshold profiles are data-dependent and must be regenerated from the frozen
  score bundle for each real release candidate;
- enrollment caches are only valid for the model-bundle metadata they were
  built against;
- no document in this repo should be read as a claim that the current
  checked-in `artifacts/model-bundle/` contents are a production speaker
  encoder.

## Deployment Notes

The active runtime paths are configured through:

- `configs/deployment/infer.toml` for CPU-oriented startup;
- `configs/deployment/infer-gpu.toml` for `gpu-server` startup.

Operational requirements:

- the service should be started with strict artifact checks for target runs via
  `--require-artifacts` or `KRYP_REQUIRE_DEPLOYMENT_ARTIFACTS=1`;
- `artifacts/model-bundle`, `artifacts/enrollment-cache`,
  `artifacts/manifests`, and the active threshold bundle under `artifacts/`
  must stay in sync;
- the demo threshold resolver scans `artifacts/` by newest file mtime, so a
  newer stale `verification_threshold_calibration.json` can silently override
  the intended threshold unless rollout is controlled carefully;
- health and demo-state inspection should always check both:
  - requested backend: `requested_backend`;
  - resolved backend: `selected_backend`;
  - real embedding implementation: `inferencer.implementation`.

Operational references:

- [deployment/README.md](../deployment/README.md)
- [web-demo.md](./web-demo.md)
- [inference-observability.md](./inference-observability.md)
- [enrollment-embedding-cache.md](./enrollment-embedding-cache.md)
- [release-runbook.md](./release-runbook.md)

## Release Checklist

Before a candidate is presented as release-ready:

1. Freeze exact training and deploy configs.
2. Freeze offline quality and threshold artifacts.
3. Run serving smoke, regression, and stress workflows on the intended runtime.
4. Rebuild or verify the enrollment cache against the active model bundle.
5. Build the final benchmark pack and use it as the review artifact.
6. Record known scope limits and rollback inputs in the runbook.
7. Build the submission/release bundle so the handoff contains the frozen
   docs, configs, model artifacts, thresholds, and demo assets in one place.
