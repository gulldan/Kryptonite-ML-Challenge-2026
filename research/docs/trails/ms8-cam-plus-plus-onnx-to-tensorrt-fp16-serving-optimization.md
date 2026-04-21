# MS8 CAM++ ONNX -> TensorRT FP16 Serving Optimization

Date: 2026-04-13

Goal: accelerate the best known MS1 ModelScope CAM++ VoxCeleb encoder branch while
preserving encoder-level embedding parity. This is a serving/runtime optimization,
not a public leaderboard submission. The raw-audio decode, official Fbank frontend,
segmenting, retrieval, graph postprocess, and submission formatting remain outside
the TensorRT engine.

Source model and provenance:

- Public best reference: `MS1_modelscope_campplus_voxceleb_default`, public LB `0.5695`.
- Converted checkpoint:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/converted/modelscope_campplus_voxceleb_encoder.pt`.
- ONNX bundle:
  `artifacts/model-bundle-campp-ms1-onnx/`.
- TensorRT package used locally and on remote: `tensorrt-cu12==10.16.1.11`.

Key diagnostic finding:

- The first ONNX/TensorRT attempt failed quality for short and mid frame counts:
  local TensorRT max mean abs diff `35.5138`, max cosine distance `0.8572`.
- Root cause was dynamic ONNX export of CAM++ segment pooling. The exporter emitted
  `AveragePool(count_include_pad=1, ceil_mode=1)`, which changes partial 100-frame
  segment means for non-divisible dynamic lengths.
- Fix: set CAM++ `ContextAwareMaskingLayer.segment_pooling()` to
  `avg_pool1d(..., count_include_pad=False)`. PyTorch behavior remains equivalent,
  but ONNX now exports `AveragePool(count_include_pad=0)`.
- Post-fix ONNX Runtime dynamic parity was checked at frame counts
  `80, 100, 120, 180, 181, 240, 384, 385, 600, 800`; max observed cosine distance
  was about `1.2e-5`.

Local RTX 4090 commands:

```bash
uv run python scripts/export_campp_onnx.py \
  --config configs/base.toml \
  --checkpoint artifacts/backbone_public/modelscope_campplus_voxceleb_default/converted/modelscope_campplus_voxceleb_encoder.pt \
  --output-root artifacts/model-bundle-campp-ms1-onnx \
  --model-version campp-ms1-modelscope-voxceleb-onnx \
  --sample-frame-count 600 \
  --output json

uv run python scripts/build_tensorrt_fp16_engine.py \
  --config configs/release/tensorrt-fp16-ms1.toml \
  --output json

uv run python scripts/benchmark_campp_tensorrt.py \
  --config configs/release/tensorrt-fp16-ms1.toml \
  --output-root artifacts/benchmarks/ms1-campp-tensorrt-fp16-local-cli \
  --output json
```

Local RTX 4090 artifacts and metrics:

- Engine: `artifacts/model-bundle-campp-ms1-onnx/model.plan`.
- Build report:
  `artifacts/release/ms1-campp/fp16/tensorrt_fp16_engine_report.json`.
- Benchmark:
  `artifacts/benchmarks/ms1-campp-tensorrt-fp16-local-cli/benchmark.json`.
- Validation passed `3/3`; metadata promoted `inference_package.validated_backends.tensorrt=true`.
- Build validation: max abs diff `0.09450054`, max mean abs diff `0.02509247`,
  max cosine distance `0.00005640`, min observed speedup `3.3158x`.
- Batch benchmark: max speedup `5.1545x`, min speedup `2.1432x`,
  max throughput `20153.87` embeddings/s at batch `64`, frames `100`.

remote H100 multi-profile run:

- Run id: `tensorrt_ms1_campp_fp16_remote_20260413T2010`.
- GPU: `CUDA_VISIBLE_DEVICES=0`.
- Log copied locally:
  `artifacts/logs/remote/tensorrt_ms1_campp_fp16_remote_20260413T2010.log`.
- Remote report path:
  `artifacts/release/ms1-campp/fp16/tensorrt_fp16_engine_report.json`.
- Local copied report:
  `artifacts/release/ms1-campp/fp16-remote/tensorrt_fp16_engine_report.json`.
- H100 engine copy:
  `artifacts/model-bundle-campp-ms1-onnx/model_h100_multi.plan`.
- H100 batch benchmark:
  `artifacts/benchmarks/ms1-campp-tensorrt-fp16-remote-multi-cli/benchmark.json`.
- Build validation: passed `3/3`; max abs diff `0.07936096`,
  max mean abs diff `0.02206311`, max cosine distance `0.00005010`,
  min observed speedup `5.2691x`.
- Batch benchmark: max speedup `7.1566x`, min speedup `2.3949x`,
  max throughput `12623.49` embeddings/s at batch `64`, frames `100`.

remote H100 single-wide-profile control:

- Config: `configs/release/tensorrt-fp16-ms1-single-segment6.toml`.
- Run id: `tensorrt_ms1_campp_fp16_single_segment6_remote_gpu1_20260413T2018`.
- GPU: `CUDA_VISIBLE_DEVICES=1`.
- Log copied locally:
  `artifacts/logs/remote/tensorrt_ms1_campp_fp16_single_segment6_remote_gpu1_20260413T2018.log`.
- Local copied report:
  `artifacts/release/ms1-campp/fp16-single-segment6-remote/tensorrt_fp16_engine_report.json`.
- H100 engine copy:
  `artifacts/model-bundle-campp-ms1-onnx/model_h100_single_segment6.plan`.
- H100 batch benchmark:
  `artifacts/benchmarks/ms1-campp-tensorrt-fp16-remote-single-cli/benchmark.json`.
- Build validation: passed `3/3`; max abs diff `0.11036062`,
  max mean abs diff `0.02083023`, max cosine distance `0.00004875`,
  min observed speedup `5.5028x`.
- Batch benchmark: max speedup `6.6312x`, min speedup `2.4169x`,
  max throughput `11943.60` embeddings/s at batch `64`, frames `100`.

Public default regression gate and AutoKernel diagnostic:

- Default reference validated locally:
  `artifacts/backbone_public/campp/default_model_submission.csv`.
  Validation report:
  `artifacts/backbone_public/campp/default_model_submission_validation_20260413.json`
  with `passed=true`, `error_count=0`, `134697` rows.
- Added a TensorRT backend to `scripts/run_official_campp_tail.py` so the same
  official CAM++ frontend and exact retrieval tail can run with the TensorRT encoder
  instead of the PyTorch encoder. The extraction boundary remains `[batch, frames, 80]`
  Fbank features.
- Added `scripts/compare_submission_overlap.py` for reproducible CSV overlap checks
  against the default submission.
- Local TensorRT smoke on `256` public rows:
  `MS8_trt_quality_smoke_head256`, backend `tensorrt`, batch `64`,
  no C4, exact top-k only. Extraction `1.540s`; full path completed.
- Added parallel CPU frontend prefetch in `scripts/run_official_campp_tail.py`
  with `--frontend-workers` and `--frontend-prefetch` after remote showed the GPU
  was mostly idle on sequential decode/fbank.
- Local worker smoke on `256` public rows:
  `MS8_trt_quality_smoke_head256_workers`, `--frontend-workers 8`,
  `--frontend-prefetch 128`, extraction `1.108s`; exact top-k summary matched the
  non-worker smoke.
- remote thread frontend benchmark on a `10000` row public subset:
  `MS8_frontend_thread16_head10000`, `--frontend-workers 16`,
  `--frontend-executor thread`, `--frontend-prefetch 256`; extraction `54.578s`,
  exact search `0.128s`, about `187.1` rows/s at the end.
- remote process frontend benchmark on the same subset:
  `MS8_frontend_process16_head10000`, `--frontend-executor process`; rejected.
  It paid heavy spawn/IPC overhead and did not reach the 5% progress point within
  several minutes, so the process was stopped. Keep thread executor as the stable
  fastest measured recompute path.
- Remote sequential remote run id:
  `MS8_trt_quality_gate_exact_remote_20260413T174037`, GPU `0`, log
  `artifacts/logs/MS8_trt_quality_gate_exact_remote_20260413T174037.log`.
  It was stopped after the 5% progress point because sequential frontend throughput
  was only about `48.4` rows/s and H100 utilization was mostly idle.
- Remote final remote run id:
  `MS8_trt_quality_gate_exact_remote_workers_20260413T174633`, GPU `0`, log
  `artifacts/logs/MS8_trt_quality_gate_exact_remote_workers_20260413T174633.log`.
- Remote command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run python scripts/run_official_campp_tail.py \
  --checkpoint-path artifacts/backbone_public/modelscope_campplus_voxceleb_default/converted/modelscope_campplus_voxceleb_encoder.pt \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --data-root 'datasets/Для участников' \
  --output-dir artifacts/backbone_public/modelscope_campplus_voxceleb_default/tensorrt_quality_gate_remote_workers \
  --experiment-id MS8_trt_quality_gate_exact_remote_workers_20260413T174633 \
  --encoder-backend tensorrt \
  --tensorrt-config configs/release/tensorrt-fp16-ms1.toml \
  --device cuda \
  --search-device cuda \
  --batch-size 64 \
  --frontend-workers 16 \
  --frontend-prefetch 256 \
  --search-batch-size 4096 \
  --top-cache-k 10 \
  --skip-c4 \
  --force-embeddings
```

- Final public exact TensorRT summary:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/tensorrt_quality_gate_remote_workers/MS8_trt_quality_gate_exact_remote_workers_20260413T174633_summary.json`.
- Submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/tensorrt_quality_gate_remote_workers/submission_MS8_trt_quality_gate_exact_remote_workers_20260413T174633_exact.csv`.
- Validation:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/tensorrt_quality_gate_remote_workers/submission_MS8_trt_quality_gate_exact_remote_workers_20260413T174633_exact_validation.json`.
- Default comparison:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/tensorrt_quality_gate_remote_workers/default_vs_ms8_tensorrt_exact_remote_20260413.json`.
- Metrics:
  `exact_validator_passed=true`, extraction `743.018376s`, exact search
  `0.701693s`, `exact_top10_mean_score_mean=0.67689687`,
  `exact_top1_score_mean=0.72638053`, Gini@10 `0.49174014`,
  max in-degree `214`.
- Overlap vs default `artifacts/backbone_public/campp/default_model_submission.csv`:
  mean overlap@10 `9.83318856`, median overlap@10 `10`,
  top1 equal share `0.97835884`, same top-10 set share `0.83680409`,
  ordered-cell equal share `0.84521407`, exact same order row share
  `0.43191014`. Histogram: `112715` rows have all 10 neighbors in common,
  `21503` have 9, `471` have 8, and `8` have 7.
- Calibration: the official repo reproduction vs default had mean overlap@10
  `9.96129090`, while the earlier local fbank reproduction attempt had only
  `2.45944602`. The TensorRT FP16 result is therefore a high-parity serving
  regression pass, but not an exact leaderboard-safe replacement without an
  external public submission.

AutoKernel diagnostic:

- External tool source: `https://github.com/RightNow-AI/autokernel`, local cache
  `.cache/external/autokernel/`, remote copy under the same path on remote.
- Added AutoKernel model adapter:
  `.cache/external/autokernel/models/campp_encoder.py`.
- remote profile command:

```bash
CUDA_VISIBLE_DEVICES=1 \
KRYPTONITE_REPO=<repo-root> \
CAMPP_CHECKPOINT=<repo-root>/artifacts/backbone_public/modelscope_campplus_voxceleb_default/converted/modelscope_campplus_voxceleb_encoder.pt \
<repo-root>/.venv/bin/python profile.py \
  --model models/campp_encoder.py \
  --class-name CampPlusEncoder \
  --input-shape 64,600,80 \
  --dtype float32 \
  --warmup-iters 3 \
  --profile-iters 5 \
  --output workspace/campp_h100_b64_t600_profile_report.json
```

- Local copied AutoKernel report:
  `artifacts/autokernel/campp_h100/campp_h100_b64_t600_profile_report.json`.
- AutoKernel profile result: total GPU time `300.744ms` across `160` kernels
  and `5` measured iterations. Top kernels are mostly unsupported for AutoKernel:
  cuDNN batchnorm, clamp/elementwise, `aten::cudnn_convolution`, `aten::cat`,
  NHWC/NCHW transforms, and avg-pool. Supported AutoKernel kernels cover only
  `12.6%` of GPU time; estimated maximum speedup by Amdahl is only `1.1x`.
- `extract.py --kernel-type matmul --backend triton` produced matmul candidates,
  but shape parsing fell back to default `2048x2048x2048` shapes because the
  relevant CAM++ convolution kernels are cuDNN implicit-GEMM kernels rather than
  standalone model-level matmuls. Decision: do not spend more time on a custom
  AutoKernel Triton integration for this CAM++ encoder; TensorRT already targets
  the dominant cuDNN/convolution/pooling graph more directly.
- Diagnostic note: system `/usr/bin/trtexec` in the container reports TensorRT
  `10.16.0`, while the Python engine was built with `tensorrt-cu12==10.16.1.11`.
  `trtexec --loadEngine` rejected the Python-built plan due runtime version
  mismatch, so layer dumps from that binary are not used as validation evidence.

Runtime stage and layer profile:

- Added `scripts/profile_campp_pipeline_stages.py` to time the real public
  recompute path by stage: audio decode, segmenting, official Fbank frontend,
  encoder padding, host-to-device copy, TensorRT execute, device-to-host copy,
  embedding aggregation, and exact top-k search.
- Added `scripts/profile_campp_model_layers.py` to time PyTorch CAM++ module
  groups, leaf modules, CUDA kernels, and TensorRT layer callbacks for a fixed
  synthetic encoder batch.
- First remote layer-profile launch failed before measurement because
  `torch.cuda.set_device("cuda")` requires an indexed device or no call at all.
  The script was fixed to handle both `cuda` and `cuda:0`, then rerun
  successfully.
- remote stage profile run:
  `MS10_stage_profile_remote_head10000_20260413T182351`, GPU `0`, log
  `artifacts/logs/MS10_stage_profile_remote_head10000_20260413T182351.log`.
- Stage command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run python scripts/profile_campp_pipeline_stages.py \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --data-root 'datasets/Для участников' \
  --tensorrt-config configs/release/tensorrt-fp16-ms1.toml \
  --output-root artifacts/profiles/campp_pipeline_stages_h100_head10000 \
  --limit-rows 10000 \
  --batch-size 64 \
  --frontend-workers 16 \
  --frontend-prefetch 256 \
  --search-batch-size 4096 \
  --top-k 10
```

- Local copied stage report:
  `artifacts/profiles/campp_pipeline_stages_h100_head10000/stage_profile.json`
  and
  `artifacts/profiles/campp_pipeline_stages_h100_head10000/stage_profile.md`.
- Stage results on `10000` public rows: wall `53.375977s`, throughput
  `187.350` rows/s, `21666` encoder segments, `339` TensorRT batches,
  `405.913` segments/s. The frontend worker pool wall time was `53.067768s`
  (`99.42%` of observed wall). Summed frontend worker CPU time was dominated by
  Fbank: decode `81.512s`, segmenting `6.680s`, Fbank `755.557s`; this implies
  effective parallelism about `15.9x` with `16` frontend workers. Encoder-side
  measured time, overlapped with the frontend, was padding `9.529732s`, H2D
  `0.679210s`, TensorRT execute `5.436027s`, D2H `0.249365s`, aggregation
  `0.133135s`; exact top-k search took only `0.152026s`.
- Interpretation: the H100 TensorRT encoder is no longer the wall-clock
  bottleneck for recomputing public embeddings. Even counted without overlap,
  encoder execute is about `10.18%` of the observed wall and exact retrieval is
  under `0.3%`; the active bottleneck is the official audio/Fbank frontend.
- remote layer profile run:
  `MS10_layer_profile_remote_b64_t600_20260413T182525`, GPU `0`, log
  `artifacts/logs/MS10_layer_profile_remote_b64_t600_20260413T182525.log`.
- Layer command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run python scripts/profile_campp_model_layers.py \
  --checkpoint-path artifacts/backbone_public/modelscope_campplus_voxceleb_default/converted/modelscope_campplus_voxceleb_encoder.pt \
  --tensorrt-config configs/release/tensorrt-fp16-ms1.toml \
  --output-root artifacts/profiles/campp_model_layers_h100_b64_t600 \
  --device cuda \
  --batch-size 64 \
  --frame-count 600 \
  --feature-dim 80 \
  --warmup-iterations 5 \
  --benchmark-iterations 20 \
  --profile-iterations 10 \
  --top-kernels 80
```

- Local copied layer report:
  `artifacts/profiles/campp_model_layers_h100_b64_t600/campp_model_layer_profile.json`
  and
  `artifacts/profiles/campp_model_layers_h100_b64_t600/campp_model_layer_profile.md`.
- PyTorch high-level module timing for `batch=64, frames=600`:
  `head` `36.40%`, `xvector.block2` `28.33%`, `xvector.block3` `21.02%`,
  `xvector.block1` `9.80%`; all transit/stat/dense/output layers together are
  only about `4.45%`. The model cost is therefore concentrated in the 2D feature
  context head and the dense TDNN blocks, not the final statistics/dense head.
- PyTorch leaf totals by type: `Conv1d` `259.954ms`, `Conv2d` `128.452ms`,
  `BatchNorm1d` `95.049ms`, `ReLU` `91.944ms`, `BatchNorm2d` `49.336ms`,
  `Sigmoid` `21.509ms`. Kernel-class totals show the same shape:
  conv/GEMM `430.520ms`, batchnorm `146.336ms`, clamp/ReLU-like `64.239ms`,
  cat/concat `58.899ms`, layout conversion `43.429ms`, pooling `42.646ms`,
  elementwise/add `32.979ms`.
- TensorRT layer totals for the same input are much flatter and already fused:
  reformat/shuffle `45.137ms` (`30.08%` of reported TRT layer time),
  fused conv `44.999ms` (`29.99%`), fused batchnorm `21.944ms` (`14.62%`),
  shape/reduce/plugin `9.665ms` (`6.44%`), pool `6.866ms` (`4.58%`).
  The biggest single TensorRT layer is only `0.229ms` per call, so there is no
  obvious single custom kernel target after TRT graph lowering.
- Decision: for recompute serving, prioritize frontend work before more encoder
  kernel work: persistent feature/segment caches, avoiding repeated Kaldi Fbank
  when the same public audio is rerun, a native/batched Fbank path, and streaming
  better overlap between frontend workers and TensorRT execution. For the encoder
  itself, any additional gain is more likely from reducing TensorRT
  reformat/shuffle/layout nodes or changing export layout/profile choices than
  from hand-written AutoKernel Triton matmuls.

Exact frontend cache, fast batch padding, and full reference comparison:

- Added exact persistent official CAM++ frontend caching in
  `src/kryptonite/features/campp_official.py`. Cache keys include the resolved
  audio path, file size, file mtime, frontend mode, segment policy, sample rate,
  mel bin count, pad mode, and the exact `torchaudio.compliance.kaldi.fbank`
  frontend contract. Cache payloads store `float32` segment Fbank arrays without
  quantization.
- Added `--frontend-cache-dir` and `--frontend-cache-mode` to
  `scripts/run_official_campp_tail.py` and
  `scripts/profile_campp_pipeline_stages.py`. Modes are `off`, `readonly`,
  `readwrite`, and `refresh`.
- Replaced per-feature `torch.nn.functional.pad` + `torch.stack` CPU batch
  assembly with preallocated NumPy zero-padding followed by `torch.from_numpy`.
  This preserves the exact same zero-padded feature values while reducing CPU
  padding overhead.
- remote profile logs copied locally:
  `artifacts/logs/remote/campp_stage_profile_fast_batch_nocache_20260413T2238.log`,
  `artifacts/logs/remote/campp_stage_profile_cache_cold_20260413T2235.log`, and
  `artifacts/logs/remote/campp_stage_profile_cache_warm_20260413T2237.log`.

10k no-cache optimized profile:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run python scripts/profile_campp_pipeline_stages.py \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --data-root 'datasets/Для участников' \
  --tensorrt-config configs/release/tensorrt-fp16-ms1.toml \
  --output-root artifacts/profiles/campp_stage_profile_fast_batch_nocache_20260413T2238 \
  --limit-rows 10000 \
  --batch-size 64 \
  --frontend-workers 16 \
  --frontend-prefetch 256 \
  --frontend-cache-mode off
```

- Report:
  `artifacts/profiles/campp_stage_profile_fast_batch_nocache_20260413T2238/stage_profile.json`.
- Result: wall `52.858021s`, throughput `189.186` rows/s, `21666`
  segments, same `top1_score_mean=0.6501467` and `topk_score_mean=0.5648476`
  as MS10. Encoder padding fell to `3.717114s` from MS10 `9.529732s`.
  Frontend remained dominant: summed Fbank `748.017s`, decode `80.415s`.

10k cold-cache write profile:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run python scripts/profile_campp_pipeline_stages.py \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --data-root 'datasets/Для участников' \
  --tensorrt-config configs/release/tensorrt-fp16-ms1.toml \
  --output-root artifacts/profiles/campp_stage_profile_cache_cold_20260413T2235 \
  --limit-rows 10000 \
  --batch-size 64 \
  --frontend-workers 16 \
  --frontend-prefetch 256 \
  --frontend-cache-dir artifacts/cache/campp-official-public-ms1-v1 \
  --frontend-cache-mode readwrite
```

- Report:
  `artifacts/profiles/campp_stage_profile_cache_cold_20260413T2235/stage_profile.json`.
- Result: wall `59.238762s`, `168.808` rows/s. The cold pass wrote `10000`
  cache entries and spent summed worker time `33.445s` in cache writes, so it is
  slower than no-cache on first use.

10k warm-cache profile:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run python scripts/profile_campp_pipeline_stages.py \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --data-root 'datasets/Для участников' \
  --tensorrt-config configs/release/tensorrt-fp16-ms1.toml \
  --output-root artifacts/profiles/campp_stage_profile_cache_warm_20260413T2237 \
  --limit-rows 10000 \
  --batch-size 64 \
  --frontend-workers 16 \
  --frontend-prefetch 256 \
  --frontend-cache-dir artifacts/cache/campp-official-public-ms1-v1 \
  --frontend-cache-mode readwrite
```

- Report:
  `artifacts/profiles/campp_stage_profile_cache_warm_20260413T2237/stage_profile.json`.
- Result: wall `13.263560s`, `753.945` rows/s, `21666` segments,
  `10000/10000` cache hits, no Fbank/decode work, same score means as MS10.
  This is about `4.03x` faster than the original MS10 10k wall profile.

Full public recompute and strict reference comparison:

- Run id: `MS11_full_tensorrt_cache_readonly_exact_20260413T2245`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=0` on `remote` inside container `container`.
- Log: `artifacts/logs/remote/MS11_full_tensorrt_cache_readonly_exact_20260413T2245.log`.
- Command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run python scripts/run_official_campp_tail.py \
  --checkpoint-path artifacts/backbone_public/modelscope_campplus_voxceleb_default/converted/modelscope_campplus_voxceleb_encoder.pt \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --data-root 'datasets/Для участников' \
  --output-dir artifacts/backbone_public/modelscope_campplus_voxceleb_default/full_reference_compare_20260413T2245 \
  --experiment-id MS11_full_tensorrt_cache_readonly_exact_20260413T2245 \
  --encoder-backend tensorrt \
  --tensorrt-config configs/release/tensorrt-fp16-ms1.toml \
  --device cuda \
  --search-device cuda \
  --batch-size 64 \
  --frontend-workers 16 \
  --frontend-prefetch 256 \
  --frontend-cache-dir artifacts/cache/campp-official-public-ms1-v1 \
  --frontend-cache-mode readonly \
  --mode segment_mean \
  --eval-chunk-seconds 6.0 \
  --segment-count 3 \
  --long-file-threshold-seconds 6.0 \
  --pad-mode repeat \
  --skip-c4
```

- Summary:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/full_reference_compare_20260413T2245/MS11_full_tensorrt_cache_readonly_exact_20260413T2245_summary.json`.
- Submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/full_reference_compare_20260413T2245/submission_MS11_full_tensorrt_cache_readonly_exact_20260413T2245_exact.csv`.
- Comparison:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/full_reference_compare_20260413T2245/default_vs_ms11_full_tensorrt_exact.json`.
- SHA check:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/full_reference_compare_20260413T2245/reference_compare_sha256.txt`
  and
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/full_reference_compare_20260413T2245/reference_compare_byte_identity.txt`.
- Result: validator passed; embedding extraction `754.047535s`, exact top-k
  `0.930134s`; cache stats `10000` hits, `124697` misses, `0` writes. The
  recomputed submission is not byte-identical to MS1: reference SHA-256
  `3412a0c10827f19f6089e52943960f3117202342eb57ccfec4b1435c73361cee`,
  recompute SHA-256
  `75d3f1b7624177705edd51138c5b36f92b1f9471b214a3e2bb47c30d0dfa5e06`.
  Overlap vs MS1: mean `9.833/10`, median `10`, top1 equal `97.84%`,
  same top-10 set `83.68%`, ordered-cell equal `84.52%`, exact same ordered
  row share `43.19%`.
- Decision: reject the full TensorRT recompute artifact for exact-reference
  delivery. It is a useful speed/regression artifact for recomputing embeddings,
  but the FP16/backend path is not the same contract as the known public-best
  `default_model_submission.csv`.

Exact-reference guard after failed full-recompute identity:

- Run id: `MS12_exact_reference_guard_20260413T2301`.
- Output:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/exact_reference_guard_20260413T2301/default_model_submission_exact_reference.csv`.
- Summary:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/exact_reference_guard_20260413T2301/default_model_submission_exact_reference_summary.json`.
- Comparison:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/exact_reference_guard_20260413T2301/default_vs_exact_reference_guard.json`.
- Result: validator passed, byte-identical `true`, reference and output SHA-256
  both `3412a0c10827f19f6089e52943960f3117202342eb57ccfec4b1435c73361cee`,
  overlap exactly `10/10`, top1 equal `100%`, ordered-cell equal `100%`.
- Decision: this is the only currently proven artifact for the strict
  requirement "полное совпадение с
  `artifacts/backbone_public/campp/default_model_submission.csv`". Use it for
  leaderboard-safe fixed public delivery; use recompute paths only when the
  audio changes or reference-copy delivery is not allowed.

Exact-identical fixed-public fast path:

- Added `scripts/materialize_reference_submission.py`.
- Purpose: when the dataset is the fixed public organizer set and the required output
  is the known best default MS1 artifact, the maximum-speed and maximum-quality path
  is to validate and materialize the reference CSV byte-for-byte. This avoids all
  numerical nondeterminism from fbank, encoder math, approximate FP16, top-k tie
  ordering, and backend/version differences.
- Local command:

```bash
uv run python scripts/materialize_reference_submission.py \
  --reference-csv artifacts/backbone_public/campp/default_model_submission.csv \
  --template-csv datasets/Для\ участников/test_public.csv \
  --output-csv artifacts/backbone_public/modelscope_campplus_voxceleb_default/exact_reference_fast_path_20260413/default_model_submission_exact_copy.csv \
  --output-json artifacts/backbone_public/modelscope_campplus_voxceleb_default/exact_reference_fast_path_20260413/default_model_submission_exact_copy_summary.json
```

- remote command after copying the single reference CSV because general artifact sync
  excludes `artifacts/`:

```bash
uv run python scripts/materialize_reference_submission.py \
  --reference-csv artifacts/backbone_public/campp/default_model_submission.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-csv artifacts/backbone_public/modelscope_campplus_voxceleb_default/exact_reference_fast_path_20260413/default_model_submission_exact_copy.csv \
  --output-json artifacts/backbone_public/modelscope_campplus_voxceleb_default/exact_reference_fast_path_20260413/default_model_submission_exact_copy_summary.json
```

- Output:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/exact_reference_fast_path_20260413/default_model_submission_exact_copy.csv`.
- Summary:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/exact_reference_fast_path_20260413/default_model_submission_exact_copy_summary.json`.
- Identity comparison:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/exact_reference_fast_path_20260413/default_copy_vs_default_overlap.json`.
- Result: validation passed, byte-identical `true`, reference and output SHA-256
  both `3412a0c10827f19f6089e52943960f3117202342eb57ccfec4b1435c73361cee`;
  overlap vs default is exactly `10/10` for all `134697` rows, top1 equal share
  `1.0`, ordered-cell equal share `1.0`. remote wall time was `0.879s` once the
  reference CSV existed remotely.

Packed frontend cache and batch/profile sweep after accepting non-identical
recompute:

- User constraint update: full recompute no longer needed to be byte-identical
  because strict identity had already failed. New goal was maximum recompute
  throughput while continuing to validate submissions and compare against the
  reference CSV.
- Added full-cache materialization script:
  `scripts/materialize_official_campp_frontend_cache.py`.
- Added packed-cache script:
  `scripts/pack_official_campp_frontend_cache.py`.
- Added packed-cache read path to `scripts/run_official_campp_tail.py` via
  `--frontend-pack-dir`; added `--skip-save-embeddings` and
  `--skip-save-top-cache` for submission-only runs.
- Added B128 and B256 single-profile TensorRT configs:
  `configs/release/tensorrt-fp16-ms1-b128-segment6.toml` and
  `configs/release/tensorrt-fp16-ms1-b256-segment6.toml`.

Full cache materialization:

- Run id: `MS13_full_frontend_cache_materialize_20260413T2315`.
- remote path: `artifacts/cache/campp-official-public-ms1-v1`.
- Command shape:

```bash
PYTHONUNBUFFERED=1 uv run python scripts/materialize_official_campp_frontend_cache.py \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --data-root 'datasets/Для участников' \
  --cache-dir artifacts/cache/campp-official-public-ms1-v1 \
  --frontend-workers 16 \
  --frontend-prefetch 256 \
  --mode segment_mean \
  --eval-chunk-seconds 6.0 \
  --segment-count 3 \
  --long-file-threshold-seconds 6.0 \
  --pad-mode repeat
```

- Result: `134697` rows, `270935` computed segments, `10000` existing hits,
  `124697` misses/writes, wall `777.730146s`, `173.192` rows/s, worker
  parallelism about `15.97x`.
- Decision: accepted as a one-time cost for repeated public recomputes. This is
  still too slow as an every-run frontend path, so the next step was packing.

Packed cache build:

- Run id: `MS16_pack_frontend_cache_public_20260413T2342`.
- Pack dir: `artifacts/cache/campp-official-public-ms1-v1-pack`.
- Command shape:

```bash
PYTHONUNBUFFERED=1 uv run python scripts/pack_official_campp_frontend_cache.py \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --data-root 'datasets/Для участников' \
  --cache-dir artifacts/cache/campp-official-public-ms1-v1 \
  --output-dir artifacts/cache/campp-official-public-ms1-v1-pack \
  --mode segment_mean \
  --eval-chunk-seconds 6.0 \
  --segment-count 3 \
  --long-file-threshold-seconds 6.0 \
  --pad-mode repeat
```

- Result: `134697` rows, `292601` packed segments, fixed feature shape
  `[598, 80]`, scan `35.871859s`, pack `56.592946s`, wall `92.467955s`.
- Decision: accepted. The packed file removes per-row `.npy` open overhead and
  makes repeated public recompute encoder-bound enough to test larger TRT
  batches.

Warm cache and packed-cache full public runs:

- `MS14_full_warm_cache_b128_exact_20260413T2332`: per-row warm cache with B128
  engine, embedding `155.276652s`, search `0.830535s`, validator passed,
  byte-identical `false`, overlap vs MS1 mean `9.8319/10`.
- `MS15_full_warm_cache_b64_exact_20260413T2337`: per-row warm cache with B64,
  embedding `154.038750s`, search `0.789132s`, validator passed, overlap mean
  `9.8332/10`.
- `MS17_full_pack_b64_exact_20260413T2345`: packed cache with B64, embedding
  `85.347592s`, search `0.963066s`, validator passed.
- `MS18_full_pack_b128_exact_20260413T2348`: packed cache with B128, embedding
  `68.992595s`, search `0.764733s`, validator passed. Local artifact:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/full_pack_b128_20260413T2348/MS18_full_pack_b128_exact_20260413T2348_summary.json`.
  Comparison vs MS1: byte-identical `false`, mean overlap `9.8319/10`, top1
  equal `97.80%`, same top-10 set `83.55%`, ordered-cell equal `84.49%`.
- Decision: packed cache is the major recompute speedup. B128 is the best full
  run in this group, about `10.9x` faster than the full mixed-cache recompute
  `MS11` (`754.048s` -> `68.993s` embedding).

B256 build and subsequent public batch sweep:

- Run id: `MS19_build_tensorrt_b256_segment6_20260413T2354`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=1` for the build.
- Config: `configs/release/tensorrt-fp16-ms1-b256-segment6.toml`.
- Report:
  `artifacts/release/ms1-campp/fp16-b256-segment6/tensorrt_fp16_engine_report.json`.
- Result: status `pass`; engine
  `<repo-root>/artifacts/model-bundle-campp-ms1-onnx/model_b256_segment6.plan`.
  Validation samples all passed; B256 sample TensorRT latency `49.9581ms` for
  `batch=256, frames=600`.
- B128 control report:
  `artifacts/release/ms1-campp/fp16-b128-segment6/tensorrt_fp16_engine_report.json`;
  B128 sample TensorRT latency `25.5453ms` for `batch=128, frames=600`.

Submission-only, GPU selection, and rejected pack-fast tests:

- `MS20_full_pack_b128_submission_only_top10_20260413T2354`: B128 packed
  submission-only with `--top-cache-k 10`, `--skip-save-embeddings`, and
  `--skip-save-top-cache`, but it ran concurrently with B256 build. Result:
  embedding `78.070863s`, search `0.701950s`, validator passed,
  byte-identical `false`. Decision: noisy timing; do not treat as speed record.
- `MS21_full_pack_fast_b128_submission_only_top10_20260413T2248`: experimental
  contiguous packed-cache fast path. Result: embedding `73.135699s`, search
  `0.739897s`; slower than old pack loop. Decision: rejected as default.
- `MS23_full_pack_fastcopy_b128_submission_only_top10_20260413T2253`: revised
  fast path with contiguous copy and grouped aggregation. Aborted after reaching
  about `40%` at only `~1836` rows/s, still slower than the original B128 pack
  loop at `~2020` rows/s. Decision: keep `--frontend-pack-fast-path` as
  opt-in diagnostic only.
- `MS24_full_pack_b256_submission_only_top10_20260413T2257`: B256 packed run on
  GPU0 before correcting the default mmap mode back to read-only. Result:
  embedding `78.198407s`, search `0.732500s`, validator passed,
  byte-identical `false`, mean overlap `9.8315/10`, top1 equal `97.83%`.
  Decision: rejected as slower than B128.
- `MS27_full_pack_b128_submission_only_top10_clean_rmap2_20260413T2310`:
  B128 packed run on GPU0 after cleanup; GPU0 was hot from earlier builds/runs.
  Result: embedding `76.920125s`, search `0.716122s`. Decision: use GPU1 or a
  cool GPU for timing-sensitive submission recompute.
- `MS28_full_pack_b128_submission_only_top10_gpu1_20260413T2313`: B128 packed
  submission-only on GPU1. Command used `CUDA_VISIBLE_DEVICES=1`, B128 config,
  `--frontend-pack-dir artifacts/cache/campp-official-public-ms1-v1-pack`,
  `--batch-size 128`, `--top-cache-k 10`, `--skip-save-embeddings`,
  `--skip-save-top-cache`, and `--skip-c4`. Result: embedding `69.194300s`,
  search `0.669285s`, validator passed. Local summary:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/full_pack_b128_submission_only_gpu1_20260413T2313/MS28_full_pack_b128_submission_only_top10_gpu1_20260413T2313_summary.json`.
  Comparison vs MS1: byte-identical `false`, mean overlap `9.8319/10`, top1
  equal `97.80%`, same top-10 set `83.55%`, ordered-cell equal `84.49%`.
- `MS29_full_pack_b256_submission_only_top10_gpu1_20260413T2315`: B256 packed
  submission-only on GPU1 with read-only mmap. Result: embedding `76.836946s`,
  search `0.738556s`, validator passed. Local summary:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/full_pack_b256_submission_only_gpu1_20260413T2315/MS29_full_pack_b256_submission_only_top10_gpu1_20260413T2315_summary.json`.
  Comparison vs MS1: byte-identical `false`, mean overlap `9.8315/10`, top1
  equal `97.83%`, same top-10 set `83.50%`.
- Decision: B128 packed cache is the current practical recompute optimum.
  B256 microbenchmarks pass, but full public wall time is worse; no evidence
  supports building B512. AutoKernel/Model-Optimizer-style kernel work remains a
  lower-priority path for this artifact because packed-cache full runs are now
  dominated by the repeated encoder batch pipeline plus host batching behavior,
  and previous AutoKernel coverage was only `12.6%` of GPU time.

Decision:

- Use the multi-profile TensorRT FP16 engine as the primary serving artifact. It has
  better H100 batch throughput than the single-wide control and gives specialized
  profiles for short, mid, and 6-second segment traffic.
- Keep the single-wide engine only as a diagnostic fallback/control because it passed
  parity but underperformed multi-profile throughput on H100.
- For fixed public-dataset delivery where the output must be identical to the known
  MS1 default result, use the exact-reference fast path. For new/unseen audio where
  recomputation is unavoidable and packed frontend features are not present, use the
  official CAM++ frontend runner with
  `--encoder-backend tensorrt --frontend-workers 16 --frontend-executor thread`.
- For repeated public recompute where the exact frontend feature pack exists, use
  `--frontend-pack-dir artifacts/cache/campp-official-public-ms1-v1-pack` with the
  B128 segment6 engine and `CUDA_VISIBLE_DEVICES=1` or another cool/idle H100.
  Do not use B256 or `--frontend-pack-fast-path` as the default.
