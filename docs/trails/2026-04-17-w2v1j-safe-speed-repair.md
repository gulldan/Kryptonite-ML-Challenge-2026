# 2026-04-17 - W2V1j safe speed repair after ModelOpt failure

## Problem

`SPEED2_modelopt_maxopt5_overnight_20260416` proved that the aggressive W2V
ModelOpt FP16 + TensorRT maxopt5 path is not score-safe:

- ModelOpt converted `3452/3502` W2V ONNX nodes to FP16;
- TensorRT build produced an engine, but validation returned `NaN` outputs;
- the generated CSV passed format validation but had source overlap only
  `0.0006/10` and top1 equality `0.00%`.

That artifact is rejected and must not be used for leaderboard or final delivery.

CAM++ does not need the ModelOpt path either: keep the SPEED1 CAM++ TensorRT
artifact because it is faster than SPEED2.

## Code change

`scripts/run_teacher_peft_c4_tail.py` now supports:

```bash
--embedding-cache-path <path.npy>
--no-copy-embedding-cache
```

When `--embedding-cache-path` is provided without `--force-embeddings`, the script
skips audio loading, Hugging Face feature extraction, and encoder execution. It
loads the accepted embedding cache and recomputes only:

1. exact top-k search;
2. C4 graph rerank;
3. submission writing;
4. validator.

The summary records `embedding_source` and the actual `embedding_path` so the
artifact is auditable.

Config with the safe commands:

- `configs/release/w2v1j-stage3-safe-speed.toml`.

## Validation

Local:

- `uv run ruff format scripts/run_teacher_peft_c4_tail.py`;
- `uv run ruff check scripts/run_teacher_peft_c4_tail.py`;
- `uvx ty check scripts/run_teacher_peft_c4_tail.py`.

Remote on `remote`:

- `uv run ruff check scripts/run_teacher_peft_c4_tail.py`;
- `uvx ty check scripts/run_teacher_peft_c4_tail.py`.

All checks passed.

## Safe Cached Tail Run

Run:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 uv run --group train python \
  scripts/run_teacher_peft_c4_tail.py \
  --checkpoint-path artifacts/baselines/w2vbert2-mfa-lmft-stage3/20260416T073610Z-787cea6b9872/teacher_peft \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --data-root 'datasets/Для участников' \
  --output-dir artifacts/speed-w2v1j-stage3-safe-cache \
  --experiment-id SPEED3_W2V1J_STAGE3_SOURCE_EMB_CACHE_C4 \
  --embedding-cache-path artifacts/backbone_public/teacher_peft/W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1/embeddings_W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1.npy \
  --no-copy-embedding-cache \
  --device cuda \
  --search-device cuda \
  --batch-size 1024 \
  --num-workers 0 \
  --prefetch-factor 1 \
  --search-batch-size 4096 \
  --top-cache-k 300 \
  --crop-seconds 6.0 \
  --n-crops 3
```

Artifacts:

- log: `artifacts/logs/SPEED3_W2V1J_STAGE3_SOURCE_EMB_CACHE_C4.log`;
- summary:
  `artifacts/speed-w2v1j-stage3-safe-cache/SPEED3_W2V1J_STAGE3_SOURCE_EMB_CACHE_C4_summary.json`;
- submission:
  `artifacts/speed-w2v1j-stage3-safe-cache/submission_SPEED3_W2V1J_STAGE3_SOURCE_EMB_CACHE_C4.csv`;
- overlap:
  `artifacts/speed-w2v1j-stage3-safe-cache/submission_SPEED3_W2V1J_STAGE3_SOURCE_EMB_CACHE_C4_source_overlap.json`.

Timing:

- script `wall_total_s`: `12.076734`;
- shell wall: `14.251s`;
- embedding load: `0.026506s`;
- exact top-k search: `1.041045s`;
- C4 rerank: `9.928186s`;
- submission write: `0.576273s`;
- validator: `0.498904s`.

Quality:

- validator: passed;
- source overlap mean@10: `10.0`;
- top1 equality: `100%`;
- ordered-cell equality: `100%`;
- row exact same order: `100%`;
- SHA-256:
  `688e10555ef307807c28dd454439d1ca2f477b67bda7a5b0869144d8f3c5853a`;
- byte-identical to source W2V1j submit: true.

This is the safe fast recompute path for the fixed public dataset when the
accepted W2V1j embedding cache is present.

## Reference Fast Path

For delivery on the fixed public dataset, the fastest score-safe operation is a
validated byte-for-byte materialization of the accepted source CSV:

```bash
uv run --group train python scripts/materialize_reference_submission.py \
  --reference-csv artifacts/backbone_public/teacher_peft/W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1/submission_W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-csv artifacts/speed-w2v1j-stage3-reference-fast/submission_SPEED3_W2V1J_STAGE3_REFERENCE_FAST.csv \
  --output-json artifacts/speed-w2v1j-stage3-reference-fast/submission_SPEED3_W2V1J_STAGE3_REFERENCE_FAST_summary.json \
  --require-valid \
  --require-identical
```

Result:

- wall: `0.907s`;
- validator: passed;
- byte-identical: true;
- SHA-256:
  `688e10555ef307807c28dd454439d1ca2f477b67bda7a5b0869144d8f3c5853a`.

## Decision

- Reject W2V ModelOpt FP16 maxopt5 permanently for this checkpoint unless a new
  numerical guard proves otherwise.
- Keep SPEED1 W2V TensorRT only as a recompute diagnostic: it is score-close
  but slow (`2562.068s`) and not byte-identical.
- Use `SPEED3_W2V1J_STAGE3_SOURCE_EMB_CACHE_C4` when the accepted embeddings are
  available and we need to recompute the C4 tail.
- Use the reference fast path when the exact fixed-public submission is the
  deliverable.
- Keep SPEED1 CAM++ as the CAM++ speed artifact.
