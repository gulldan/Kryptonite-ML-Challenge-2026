# Submission / Release Bundle

## Goal

`KRYP-078` packages the final handoff into one self-contained directory and
optional `.tar.gz` archive so the next operator does not have to reconstruct a
release from scattered artifacts.

The bundle is intentionally broader than the benchmark pack:

- benchmark pack freezes quality, thresholds, latency, and provenance;
- submission bundle stages the artifacts that a reviewer or deploy owner
  actually needs to pick up and validate the release.

## What Goes Into The Bundle

The bundle builder stages a deterministic folder layout with:

- repository README plus release docs (`model-card`, `release-runbook`, and any
  additional documentation paths);
- deployment configs;
- active ONNX model bundle metadata;
- optional TensorRT plan;
- frozen checkpoints;
- threshold calibration JSON;
- frozen benchmark summary files;
- demo assets;
- optional Triton model repository and other supporting metadata.

It also writes:

- `README.md` with validation commands and warnings;
- `submission_bundle.json` manifest with staged paths and checksums;
- `submission_bundle.md` human-readable summary;
- `release_freeze.json` with one explicit freeze snapshot for code/data/model/engine;
- `release_freeze.md` with the same snapshot in operator-friendly Markdown;
- optional `<bundle-root>.tar.gz` archive.

## Bundle Modes

Two modes are supported:

- `smoke`: honest packaging of the current checked-in demo/runtime artifacts.
  Candidate-only inputs such as frozen thresholds or final checkpoints may be
  omitted.
- `candidate`: strict handoff mode for a real release candidate. At least one
  benchmark artifact, one threshold calibration JSON, and one checkpoint are
  required.

The builder surfaces warnings when the bundle still carries demo/stub artifacts,
for example `structural_stub = true` in `model-bundle/metadata.json`.

## Usage

Start from the checked-in template:

```bash
uv run python scripts/build_submission_bundle.py \
  --config configs/release/submission-bundle.example.toml
```

For a real candidate handoff:

1. switch `bundle_mode` to `candidate`;
2. set `release_tag` and list the exact `data_manifest_paths` that define the
   dataset freeze for this candidate;
3. replace the placeholder smoke paths with frozen benchmark/threshold/checkpoint
   artifacts;
4. set `tensorrt_plan_path` and `require_tensorrt_plan = true` if the release
   explicitly depends on the TensorRT handoff;
5. rerun the builder.

## Output Layout

One run writes:

```text
artifacts/release-bundles/<bundle-id>/
├── README.md
├── release_freeze.json
├── release_freeze.md
├── submission_bundle.json
├── submission_bundle.md
├── benchmark/
├── checkpoints/
├── configs/
├── data-manifests/
├── demo/
├── docs/
├── model/
├── supporting/
├── thresholds/
├── triton-model-repository/
└── sources/
    └── submission_bundle_config.toml
```

The archive is written alongside the directory:

```text
artifacts/release-bundles/<bundle-id>.tar.gz
```

## Validation

The generated bundle README includes a short clean-machine validation path. The
expected minimum checks are:

```bash
uv sync --dev --group train --group tracking
uv run python scripts/infer_smoke.py --config configs/deployment/infer.toml --require-artifacts
```

If the bundle includes the Triton repository, also validate it with
`scripts/triton_infer_smoke.py`.

## Release Freeze

`KRYP-079` extends the bundle with an explicit release-freeze snapshot:

- `code`: fingerprints the repository runtime surface (`pyproject.toml`,
  `uv.lock`, `src/`, `apps/`, `scripts/`, `configs/`) and records git metadata
  when available;
- `data`: stages the declared `data_manifest_paths` and records one aggregate
  checksum for the manifest freeze;
- `model`: records the frozen checkpoint and model metadata payload;
- `engine`: records the ONNX export plus optional TensorRT/Triton handoff
  artifacts.

Candidate mode now requires both `release_tag` and at least one
`data_manifest_paths` entry so the handoff always captures an explicit dataset
freeze instead of only the runtime bundle.

## Scope Limits

- The submission bundle does not invent missing release evidence.
- In `smoke` mode it is allowed to package demo/stub artifacts, but it calls
  that out explicitly in the generated warnings.
- The bundle does not replace the benchmark pack; it depends on it for frozen
  release evidence.
