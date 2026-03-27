# Final Benchmark Pack

## Goal

`KRYP-076` freezes one self-contained release bundle that can compare any two
final candidates without re-chasing scattered artifacts.

The pack captures four release signals per candidate:

- offline verification quality from `verification_eval_report.json`
- deployment thresholds from `verification_threshold_calibration.json`
- serving latency and hard limits from `inference_stress_report.json`
- process/CUDA memory peaks plus exact config and bundle provenance

## Inputs

The builder consumes a TOML config with at least two `[[candidate]]` entries.

Use the checked-in template as the starting point:

```bash
uv run python scripts/build_final_benchmark_pack.py \
  --config configs/eval/final-benchmark-pack.example.toml
```

Replace the placeholder candidate paths with frozen release artifacts before
running the command for real.

Each candidate entry points at:

- one `verification_eval_report.json`
- one optional `verification_threshold_calibration.json`
- one `inference_stress_report.json`
- one `model-bundle/metadata.json`
- one or more exact `.toml` config files
- optional supporting JSON/Markdown artifacts such as final-selection notes

## Recommended Workflow

For each candidate:

1. Generate or freeze the offline quality report.
2. Generate or freeze the threshold calibration bundle.
3. Run the serving stress report with the candidate's deploy config.
4. Add those paths plus the exact config files to the benchmark-pack TOML.
5. Build the pack.

Example release-oriented command sequence:

```bash
uv run python scripts/evaluate_verification_scores.py \
  --scores artifacts/final/campp/dev_scores.jsonl \
  --trials artifacts/final/campp/dev_trials.jsonl \
  --metadata artifacts/final/campp/dev_embedding_metadata.parquet

uv run python scripts/calibrate_verification_thresholds.py \
  --scores artifacts/final/campp/dev_scores.jsonl \
  --trials artifacts/final/campp/dev_trials.jsonl \
  --metadata artifacts/final/campp/dev_embedding_metadata.parquet

uv run python scripts/inference_stress_report.py \
  --config configs/deployment/infer-gpu.toml \
  --output-root artifacts/final/campp/stress

uv run python scripts/build_final_benchmark_pack.py \
  --config configs/eval/final-benchmark-pack.toml
```

## Output Contract

One pack run writes:

```text
artifacts/benchmark-pack/<pack-name>/
├── final_benchmark_pack.json
├── final_benchmark_pack.md
├── final_benchmark_pack_candidates.jsonl
├── final_benchmark_pack_pairwise.jsonl
├── sources/
│   └── final_benchmark_pack_config.toml
└── candidates/
    └── <candidate-id>/
        └── sources/
            ├── verification_eval_report.json
            ├── verification_threshold_calibration.json
            ├── inference_stress_report.json
            ├── model_bundle_metadata.json
            ├── export_boundary.json
            ├── config_01_<name>.toml
            └── supporting_01_<name>.json
```

The copied `sources/` tree is intentional: the pack stays readable even if the
original training/deploy directories move later.

## Pairwise Comparison Rules

The pack computes pairwise comparisons for every candidate pair and records:

- `dEER` and `dMinDCF` as `right - left`
- `dLatency ms/audio` at the largest validated burst size
- `dRSS MiB` from the stress report peak process RSS
- `dCUDA MiB` from the stress report peak CUDA allocated memory

Lower values are treated as better for all four metrics.

## Memory Notes

The stress report now records:

- peak process RSS in MiB
- peak process RSS delta in MiB
- peak CUDA allocated/reserved memory in MiB when CUDA is active

These numbers are operational release metrics, not training-memory claims.

## Scope Limits

- The pack does not invent quality metrics; it only packages frozen reports.
- The pack does not replace shortlist/model-selection workflows.
- The pack does not infer missing memory values from old stress reports; those
  fields remain empty until the candidate is re-profiled with the updated
  stress workflow.
