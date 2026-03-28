# Backend Benchmark

`KRYP-064` freezes one reproducible benchmark report for the promoted export
stack:

- `PyTorch` checkpoint inference
- `ONNX Runtime` session inference
- `TensorRT` plan inference

The workflow operates on the encoder boundary
`[batch, frames, mel_bins] -> embedding`, so it isolates backend execution from
the raw-audio frontend.

## Command

```bash
uv run python scripts/build_backend_benchmark_report.py \
  --config configs/release/backend-benchmark.toml
```

## Outputs

The report writes to `artifacts/release/current/backend-benchmark/`:

- `backend_benchmark_report.json`
- `backend_benchmark_report.md`
- `backend_benchmark_workload_rows.jsonl`
- `backend_benchmark_latency_batch*.svg`
- `sources/backend_benchmark_config.toml`

## What is measured

- cold start: backend initialization plus first inference
- warm latency: repeated timed iterations after warmup
- throughput: items/sec and frames/sec
- stability: latency coefficient of variation and p95
- memory: process RSS deltas plus optional process-local GPU memory snapshots
- quality drift: mean abs diff, max abs diff, and cosine distance versus the
  PyTorch reference output

## Notes

- The benchmark config must include at least one `batch_size=1` workload and at
  least one batched workload so the report always emits graphs for both modes.
- `ONNX Runtime` is benchmarked independently from TensorRT: the workflow
  intentionally prefers `CUDAExecutionProvider` or `CPUExecutionProvider`, not
  `TensorrtExecutionProvider`.
- For realistic numbers, run on `gpu-server` after the TensorRT engine workflow
  has produced `tensorrt_fp16_engine_report.json`.
