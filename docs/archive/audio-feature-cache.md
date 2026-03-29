# Audio Feature Cache

## Goal

Lock in one explicit policy for where the `80-dim` log-Mel frontend should live
across train, dev, and infer:

- train: precompute on CPU and reuse a disk-backed cache
- dev/eval: allow an optional cache for repeated reruns
- infer/demo: keep runtime extraction as the default path

`KRYP-026` adds both the cache implementation and the reproducible benchmark
flow that justifies this split.

## Cache Layout

Feature cache artifacts live under:

```text
artifacts/cache/features/<namespace>/<aa>/<bb>/<cache_id>.pt
```

The default namespace is `fbank-v1`, configured in `configs/base.toml`:

```toml
[feature_cache]
namespace = "fbank-v1"
train_policy = "precompute_cpu"
dev_policy = "optional"
infer_policy = "runtime"
benchmark_device = "auto"
benchmark_warmup_iterations = 1
benchmark_iterations = 3
```

## Invalidation Policy

Cache keys are content-addressed from:

- source file path, size, and `mtime`
- loader-time windowing state: start/duration, resample/downmix outcome
- VAD and loudness policy decisions
- the full Fbank request payload
- the cache format version and namespace

That means any material source-file change or frontend/config change resolves to
a different cache key automatically. Old artifacts can coexist until they are
cleaned up from `artifacts/cache/`.

## CLI Workflow

Build the cache, benchmark CPU/GPU behavior, and write a policy report from a
manifest-backed split:

```bash
uv run python scripts/feature_cache_report.py \
  --config configs/base.toml \
  --manifest artifacts/manifests/ffsvc2022-surrogate/dev_manifest.jsonl \
  --output-dir artifacts/eda/feature-cache
```

Useful flags:

- `--limit N` for fast smoke runs
- `--benchmark-device cpu|cuda|auto` to pin the benchmark target
- `--force` to rewrite existing cache entries before benchmarking

The command writes:

- `artifacts/eda/feature-cache/feature_cache_report.json`
- `artifacts/eda/feature-cache/feature_cache_report.md`
- `artifacts/eda/feature-cache/feature_cache_rows.jsonl`

## Operational Decision

The repository policy is:

- train uses CPU precompute because repeated per-epoch frontend work is wasted
  once manifests and preprocessing are stable
- dev/eval keeps cache optional because repeated ablations benefit from reuse,
  but one-shot smoke runs do not need a separate prep step
- infer keeps runtime extraction because request-scoped uploads usually do not
  have enough locality to justify cache writes and disk reads

Use the benchmark report to validate the assumption on each machine, especially
on `gpu-server` where CUDA runtime extraction may outperform cache-read plus
host-to-device transfer.
