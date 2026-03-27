# Internal Verification Protocol

`KRYP-003` freezes one repo-native verification protocol that keeps the clean
development references and the production-like robustness suites in one
auditable snapshot instead of spreading them across unrelated manifests.

## Goal

The protocol answers three questions in one place:

1. Which clean trial lists are the canonical references for development?
2. Which corrupted suites approximate the production robustness surface?
3. Which slice fields must be available before a verification report counts as
   protocol-complete?

## Protocol Shape

The checked-in config lives at `configs/eval/verification-protocol.toml` and
tracks two clean bundles:

- `official-dev-reference` for FFSVC-style public parity checks
- `speaker-disjoint-dev` for strict threshold tuning and the base trial list
  reused by production-like stress suites

The production-like side is anchored to the
`artifacts/eval/corrupted-dev-suites/corrupted_dev_suites_catalog.json` catalog,
so each robustness family stays reproducible and separately auditable:

- noise
- reverb
- codec
- distance
- channel
- silence

## Required Slices

The protocol snapshot now explicitly tracks the slice fields requested in the
ticket scope:

- `duration_bucket`
- `noise_slice`
- `rt60_slice`
- `codec_slice`
- `channel_slice`
- `distance_slice`
- `silence_ratio_bucket`
- `silence_slice`

Those fields are derived by the shared verification slice helpers in
`src/kryptonite/eval/verification_slices.py`, so the main evaluation report,
dashboard, threshold calibration, and error analysis all stay aligned.

## Builder

Render the protocol snapshot with:

```bash
uv run python scripts/build_verification_protocol.py \
  --config configs/eval/verification-protocol.toml
```

The builder writes:

- `artifacts/eval/verification-protocol/verification_protocol.json`
- `artifacts/eval/verification-protocol/verification_protocol.md`

The snapshot does not invent new trial data on its own. Instead it validates and
summarizes the clean trial lists plus the corrupted-suite catalog that upstream
builders already materialize.

## Expected Workflow

1. Regenerate the clean surrogate manifests and trial lists.
2. Rebuild the corrupted dev suites.
3. Rebuild the verification protocol snapshot.
4. Run score evaluation and threshold calibration against the relevant bundle.

This keeps the protocol deterministic, reviewable, and traceable back to the
exact manifests and suite catalogs that produced it.
