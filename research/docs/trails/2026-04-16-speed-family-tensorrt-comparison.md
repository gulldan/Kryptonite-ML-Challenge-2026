# 2026-04-16 - SPEED1: family ONNX/TensorRT speed comparison

## Goal

Build a reproducible speed comparison for one strongest model from each major
family, with the organizer baseline kept unchanged as the reference control.

The comparison must include:

- original organizer `baseline.onnx` run exactly through the organizer-style path;
- CAM++ best-family branch: MS32 encoder used by the MS41 class-aware C4 public best;
- w2v-BERT 2.0 best-family branch: W2V1j stage3 teacher-PEFT;
- ERes / WavLM best-family branch: H9 official 3D-Speaker ERes2Net-large.

Each trained family model is exported to ONNX, converted to TensorRT FP16, then
run through the same public-tail logic where possible. The final comparison uses
end-to-end timings from audio/manifest to validated submission, not only encoder
microbenchmarks.

Important measurement rule: ONNX export and TensorRT engine build are preparation
steps. They are recorded for reproducibility, but they are excluded from the speed
comparison. The speed comparison starts only after the model/engine is prepared and
measures full `submission.csv` generation: frontend, embedding extraction, search,
rerank/postprocess, CSV write, and validation.

## Selected source artifacts

| Family | Source artifact | Source public LB | Source submission |
| --- | --- | ---: | --- |
| Organizer baseline | `datasets/Для участников/baseline.onnx` | `0.0779` | organizer-provided script output |
| CAM++ | `artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt` | `0.7473` via MS41 tail | `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms41_ms32_classaware_c4_weak_20260415T0530Z/submission_MS41_ms32_classaware_c4_weak_20260415T0530Z.csv` |
| w2v-BERT 2.0 | `artifacts/baselines/w2vbert2-mfa-lmft-stage3/20260416T073610Z-787cea6b9872/teacher_peft` | `0.8344` | `artifacts/backbone_public/teacher_peft/W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1/submission_W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1.csv` |
| ERes / WavLM | `artifacts/baselines/official-3dspeaker-eres2net-large-filtered-pseudo-lowlr/20260415T040647Z-1f4c2e26a77f/official_3dspeaker_eres2net_encoder.pt` | `0.5834` | `artifacts/backbone_public/official_3dspeaker_eres2net_large_h9_pseudo_20260414T2113Z/submission_H9_official_eres_filtered_pseudo_public_c4_20260414T2113Z_c4.csv` |

Public LB numbers above are external observations from earlier uploaded files.
The TensorRT-generated submissions must be validated and compared against these
source submissions before they are treated as score-equivalent candidates.

## Repo changes for this workstream

- `scripts/export_teacher_peft_onnx.py`
  - exports W2V1j teacher-PEFT encoder from HF feature-extractor tensors to ONNX;
  - writes `metadata.json` and `export_report.md`;
  - uses ONNX external data by default for large checkpoints.
- `scripts/export_official_3dspeaker_eres2net_onnx.py`
  - exports the official 3D-Speaker ERes2Net encoder boundary `[batch, frames, 80] -> embedding`;
  - keeps the Python/Kaldi-style FBank frontend outside the engine.
- `src/kryptonite/serve/tensorrt_generic.py`
  - generic multi-input TensorRT runner and ONNX-to-engine builder.
- `scripts/build_generic_tensorrt_engine.py`
  - TOML-driven TensorRT FP16 engine build, ONNX Runtime parity check, and CUDA benchmark.
- `scripts/run_teacher_peft_c4_tail.py`
  - adds `--encoder-backend tensorrt` so W2V1j can reuse the HF feature extractor and run only the encoder in TensorRT.
- `scripts/run_official_3dspeaker_eres2net_tail.py`
  - adds `--encoder-backend tensorrt` so ERes2Net can reuse the existing FBank/chunking frontend and run only the encoder in TensorRT.
- `scripts/render_speed_comparison_chart.py`
  - renders `research/docs/assets/speed-comparison.svg` from a JSON list of completed runs.
- `configs/release/speed-family-comparison.toml`
  - records selected models, expected source submissions, commands, and remote paths.
- `configs/release/tensorrt-fp16-w2vbert2-stage3.toml`
  - W2V1j TensorRT profile and benchmark config.
- `configs/release/tensorrt-fp16-official-eres2net-h9.toml`
  - H9 ERes2Net TensorRT profile and benchmark config.

## Local validation before remote launch

Commands run locally:

```bash
uv run ruff check scripts/build_generic_tensorrt_engine.py scripts/render_speed_comparison_chart.py src/kryptonite/serve/tensorrt_generic.py scripts/export_teacher_peft_onnx.py scripts/export_official_3dspeaker_eres2net_onnx.py scripts/run_teacher_peft_c4_tail.py scripts/run_official_3dspeaker_eres2net_tail.py

uv run pytest tests/unit/test_tensorrt_engine.py tests/unit/test_onnx_export.py tests/unit/test_export_boundary.py
```

Results:

- `ruff`: passed.
- CLI help checks: passed for the two new exporters, generic TensorRT builder,
  speed chart renderer, and W2V/ERes tail scripts.
- Targeted tests: `10 passed`, `3 warnings`.

## remote execution environment

- remote host placeholder: `<remote-host>`.
- Host repository path: `<remote-repo>`.
- Container repository path: `<repo-root>`.
- Container: `container`.
- Dataset path inside container:
  `<repo-root>/datasets/Для участников`.
- Expected Python environment:
  `<repo-root>/.venv`.

Repository sync command:

```bash
rsync -a --exclude='.venv/' --exclude='.cache/' --exclude='datasets/' --exclude='artifacts/' --exclude='.pytest_cache/' --exclude='.ruff_cache/' ./ <remote-host>:<remote-repo>/
```

Runtime dependency note:

- `onnx` and `onnxruntime` are already in the train environment.
- `tensorrt` may need to be layered into the repo-local `.venv` after `uv sync`
  with `uv pip install --python .venv/bin/python tensorrt-cu12==10.16.1.11`.

## Remote run plan

All runs are launched from the container repository path with `PYTHONUNBUFFERED=1`
and logs under `artifacts/logs/`.

### CAM++ export/build/control

```bash
uv run --group train python scripts/export_campp_onnx.py \
  --config configs/base.toml \
  --checkpoint artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt \
  --output-root artifacts/model-bundle-campp-ms32-onnx \
  --model-version campp-ms32-filtered-pseudo-onnx \
  --sample-frame-count 600

uv run --group train python scripts/build_tensorrt_fp16_engine.py \
  --config configs/release/tensorrt-fp16-ms32-b128-segment6.toml
```

The broad multi-profile `configs/release/tensorrt-fp16-ms32.toml` was started first,
then stopped because the speed comparison only needs the prepared-model full-submit
path. The active CAM++ comparison engine is the single `segment6_b128` profile above.

### W2V1j export/build/public TensorRT tail

```bash
uv run --group train python scripts/export_teacher_peft_onnx.py \
  --checkpoint-path artifacts/baselines/w2vbert2-mfa-lmft-stage3/20260416T073610Z-787cea6b9872/teacher_peft \
  --output-root artifacts/model-bundle-w2vbert2-stage3-onnx \
  --model-version w2vbert2-w2v1j-stage3-onnx \
  --sample-seconds 6.0 \
  --sample-batch-size 1 \
  --opset 18

uv run --group train python scripts/build_generic_tensorrt_engine.py \
  --config configs/release/tensorrt-fp16-w2vbert2-stage3.toml

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 uv run --group train python scripts/run_teacher_peft_c4_tail.py \
  --checkpoint-path artifacts/baselines/w2vbert2-mfa-lmft-stage3/20260416T073610Z-787cea6b9872/teacher_peft \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --data-root 'datasets/Для участников' \
  --output-dir artifacts/backbone_public/teacher_peft/W2V1j_stage3_tensorrt_public_c4_20260416 \
  --experiment-id W2V1j_stage3_tensorrt_public_c4_20260416 \
  --encoder-backend tensorrt \
  --tensorrt-engine-path artifacts/model-bundle-w2vbert2-stage3-onnx/model.plan \
  --device cuda \
  --search-device cuda \
  --precision bf16 \
  --batch-size 1024 \
  --num-workers 4 \
  --prefetch-factor 1 \
  --pin-memory \
  --search-batch-size 4096 \
  --top-cache-k 300 \
  --crop-seconds 6.0 \
  --n-crops 3 \
  --force-embeddings
```

### H9 ERes2Net export/build/public TensorRT tail

```bash
uv run --group train python scripts/export_official_3dspeaker_eres2net_onnx.py \
  --checkpoint-path artifacts/baselines/official-3dspeaker-eres2net-large-filtered-pseudo-lowlr/20260415T040647Z-1f4c2e26a77f/official_3dspeaker_eres2net_encoder.pt \
  --speakerlab-root /tmp/3D-Speaker \
  --output-root artifacts/model-bundle-official-eres2net-h9-onnx \
  --model-version official-eres2net-h9-filtered-pseudo-onnx \
  --sample-frame-count 1000 \
  --opset 18

uv run --group train python scripts/build_generic_tensorrt_engine.py \
  --config configs/release/tensorrt-fp16-official-eres2net-h9.toml

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 uv run --group train python scripts/run_official_3dspeaker_eres2net_tail.py \
  --checkpoint-path artifacts/baselines/official-3dspeaker-eres2net-large-filtered-pseudo-lowlr/20260415T040647Z-1f4c2e26a77f/official_3dspeaker_eres2net_encoder.pt \
  --speakerlab-root /tmp/3D-Speaker \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --data-root 'datasets/Для участников' \
  --output-dir artifacts/backbone_public/official_3dspeaker_eres2net_large_h9_tensorrt_public_c4_20260416 \
  --experiment-id H9_official_eres_tensorrt_public_c4_20260416 \
  --encoder-backend tensorrt \
  --tensorrt-engine-path artifacts/model-bundle-official-eres2net-h9-onnx/model.plan \
  --device cuda \
  --search-device cuda \
  --batch-size 128 \
  --precision fp32 \
  --search-batch-size 4096 \
  --top-cache-k 200 \
  --chunk-seconds 10.0 \
  --force-embeddings
```

## Required comparisons after each TensorRT submit

For every TensorRT-generated CSV:

1. Run `scripts/validate_submission.py` against `datasets/Для участников/test_public.csv`.
2. Compare against the original source submission for that family with
   `scripts/compare_submission_overlap.py`.
3. Record:
   - validator status;
   - SHA-256 of the TensorRT CSV;
   - overlap@10 mean/median;
   - top1 equality share;
   - exact row-order equality share when available;
   - source public LB score and whether the TensorRT file is submitted externally.
4. Add completed timing rows to `artifacts/speed-family-comparison/speed_results.json`.
5. Regenerate:

```bash
uv run python scripts/render_speed_comparison_chart.py
```

## Current status

Status at local handoff before remote launch:

- Scripts/configs are ready and locally checked.
- Per operator constraint, all remote jobs for this workstream are run on
  `remote` `GPU1` only (`CUDA_VISIBLE_DEVICES=1`).
- TensorRT dependency was layered into the repo-local remote `.venv` with
  `uv pip install --python .venv/bin/python tensorrt-cu12==10.16.1.11`.
- Remote export/build run:
  `SPEED1_export_build_gpu1_b128_20260416T175817Z`.
- Remote export/build log:
  `artifacts/logs/SPEED1_export_build_gpu1_b128_20260416T175817Z.log`.
- Remote export/build script:
  `artifacts/logs/SPEED1_export_build_gpu1_b128_20260416T175817Z.sh`.
- First successful observed step:
  CAM++ ONNX export passed with max abs diff `0.00001764` and mean abs diff
  `0.00000474`; TensorRT build was still running when this note was written.
- The first CAM++ TensorRT build attempted the broad multi-profile
  `configs/release/tensorrt-fp16-ms32.toml`; it was stopped and replaced with
  `configs/release/tensorrt-fp16-ms32-b128-segment6.toml` because only the
  prepared-model full-submit segment6 path belongs in the speed comparison.
- CAM++ single-profile preparation then passed:
  `artifacts/model-bundle-campp-ms32-onnx/model_b128_segment6.plan`,
  report `artifacts/release/ms32-campp/fp16-b128-segment6/tensorrt_fp16_engine_report.json`.
- The first W2V1j ONNX export failed under PyTorch 2.10's default dynamo exporter:
  `dynamic_axes` could not be converted to `dynamic_shapes` for the two-input
  feature-extractor boundary. The exporter scripts were updated to force the
  legacy ONNX path with `dynamo=False`, with fallback for older PyTorch versions.
- Follow-up W2V/ERes preparation run:
  `SPEED1_w2v_eres_export_build_gpu1_20260416T180952Z`.
- Follow-up W2V/ERes preparation log:
  `artifacts/logs/SPEED1_w2v_eres_export_build_gpu1_20260416T180952Z.log`.
- W2V1j ONNX export passed on the follow-up run with mean abs diff
  `0.00000041`.
- The first W2V1j TensorRT build failed during tactic selection after skipping a
  tactic that requested about `8.6GB`, slightly above the previous `8192 MiB`
  workspace limit. Config `configs/release/tensorrt-fp16-w2vbert2-stage3.toml`
  was raised to `workspace_size_mib = 32768` for the next build attempt.
- W2V retry with larger workspace:
  `SPEED1_w2v_build32g_eres_export_build_gpu1_20260416T181214Z`.
- W2V retry log:
  `artifacts/logs/SPEED1_w2v_build32g_eres_export_build_gpu1_20260416T181214Z.log`.
- TensorRT-generated public CSVs below are validator-clean. They are not treated as
  public-LB score-equivalent unless separately submitted; source public LB values
  remain external observations from the original submitted files.

## Final remote results

All successful full-submit timings below were run on `remote` inside `container` with
`CUDA_VISIBLE_DEVICES=1`. The measured speed scope is prepared-model public
`submission.csv` generation; ONNX export and TensorRT engine build are excluded.

| Family | Run id / log | Submit generation | Main stage timing | Validator | Source overlap |
| --- | --- | ---: | --- | --- | --- |
| Organizer baseline | `SPEED1_FULL_SUBMIT_GPU1_20260416T1838Z`; `artifacts/logs/SPEED1_FULL_SUBMIT_GPU1_20260416T1838Z_organizer_baseline_baseline_public_original.log` | `254.828s` | organizer inference+indexing `254.828s` | passed | n/a |
| CAM++ MS32 TensorRT + MS41 tail | `SPEED1_FULL_SUBMIT_GPU1_FASTPACK_20260416T1848Z`; `artifacts/logs/SPEED1_FULL_SUBMIT_GPU1_FASTPACK_20260416T1848Z_campp_public_tensorrt_tail.log` | `80.486s` | embedding/frontend `67.402s`, search `0.775s`, C4 rerank `11.690s`, write `0.619s` | passed | mean@10 `8.595`, median `9`, top1 `87.28%`, row same-set `44.88%` |
| W2V1j stage3 TensorRT | `SPEED1_FULL_SUBMIT_GPU1_FASTPACK_20260416T1848Z`; `artifacts/logs/SPEED1_FULL_SUBMIT_GPU1_FASTPACK_20260416T1848Z_w2vbert2_public_tensorrt_tail.log` | `2562.068s` | embedding/frontend `2549.060s`, search `0.708s`, C4 rerank `11.653s`, write `0.647s` | passed | mean@10 `9.880`, median `10`, top1 `98.94%`, row same-set `92.99%` |
| Official ERes2Net H9 TensorRT | `SPEED1_ERES_RETRY_GPU1_WORKERS_20260416T1936Z`; `artifacts/logs/SPEED1_ERES_RETRY_GPU1_WORKERS_20260416T1936Z_eres_wavlm_public_tensorrt_tail.log` | `345.641s` | embedding/frontend `332.742s`, search `0.741s`, C4 rerank `11.532s`, write `0.625s` | passed | mean@10 `7.345`, median `8`, top1 `69.12%`, row same-set `12.01%` |

Result artifacts:

- consolidated JSON:
  `artifacts/speed-family-comparison/speed_results.json`;
- README chart:
  `research/docs/assets/speed-comparison.svg`;
- CAM++ TensorRT CSV:
  `artifacts/speed-family-comparison/campp_ms32_tensorrt/submission_SPEED_CAMPP_MS32_TRT_PUBLIC_c4.csv`;
- W2V TensorRT CSV:
  `artifacts/speed-family-comparison/w2vbert2_stage3_tensorrt/submission_SPEED_W2V1J_STAGE3_TRT_PUBLIC.csv`;
- ERes TensorRT CSV:
  `artifacts/speed-family-comparison/official_eres2net_h9_tensorrt/submission_SPEED_H9_ERES_TRT_PUBLIC_c4.csv`.

Diagnostic notes:

- The first full suite run used the organizer baseline successfully, then CAM++ was
  stopped because it was using the cold frontend path instead of the declared
  prepared packed frontend path.
- The valid CAM++ timing is the fastpack rerun above.
- The first ERes retry failed immediately because `speakerlab.process.processor.FBank`
  imports augmentation code that requires `scipy`. The runner was updated to use a
  local inference-only Kaldi FBank wrapper backed by
  `torchaudio.compliance.kaldi.fbank`, then rerun only for `eres_wavlm`.
- `nvidia.modelopt` was not available in this container. TensorRT builder/runtime,
  FP16 engines, frontend workers, and packed frontend storage are the actual
  optimizers used in the recorded runs.

## Layer profiles

Per-layer profiles were collected with `scripts/profile_tensorrt_engine_layers.py`
using the Python TensorRT runtime, because `/usr/bin/trtexec` could not deserialize
the Python-built engine files in this container.

| Family | Profile JSON | Layer mean total | Top layer family |
| --- | --- | ---: | --- |
| CAM++ | `artifacts/speed-family-comparison/layer-profiles/campp_ms32_b128_segment6_profile.json` | `24.749ms` / batch `128` | fused Conv/add/ReLU blocks, largest single layer `0.454ms` |
| W2V1j | `artifacts/speed-family-comparison/layer-profiles/w2vbert2_stage3_b1024_crop6_profile.json` | `2638.536ms` / batch `1024` | repeated fused attention blocks around `26ms` each |
| ERes2Net H9 | `artifacts/speed-family-comparison/layer-profiles/official_eres2net_h9_b128_chunk10_profile.json` | `109.217ms` / batch `128` | early ERes2Net residual Conv/Add/ReLU blocks around `3.5ms` |

Decision:

- CAM++ is the fastest prepared full-submit path in this comparison (`80.486s`).
- W2V1j remains the strongest public-LB source model (`0.8344`) but is much slower
  end-to-end because the HF audio feature extraction dominates the full submit.
- ERes2Net H9 benefits strongly from the parallel inference-only FBank frontend
  (`345.641s` vs the older source summary embedding time of about `2892s`), but
  the TensorRT CSV diverges materially from the original source submit and should
  not be treated as score-equivalent without a public submission.

## Public LB follow-up

The CAM++ SPEED1 TensorRT CSV was submitted to the public leaderboard after the
speed run:

- submitted file:
  `artifacts/submissions/SPEED_CAMPP_MS32_TRT_PUBLIC_c4_submission.csv`;
- source remote file:
  `artifacts/speed-family-comparison/campp_ms32_tensorrt/submission_SPEED_CAMPP_MS32_TRT_PUBLIC_c4.csv`;
- SHA-256:
  `632159c03085a200abf76511493dcbb53f359fb2421af30f3de14358b986ca30`;
- validator:
  passed, `134697` rows, `k=10`, errors `0`;
- public LB:
  `0.7381`;
- comparison:
  `-0.0092` vs source MS41 `0.7473`, `+0.0002` vs MS32 `0.7379`;
- decision:
  reject as a score-preserving MS41 replacement, keep as the fastest validated
  recompute artifact for the CAM++ branch.
