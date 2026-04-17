# ERes2NetV2 Remote H100 Training Setup

Date: 2026-04-11

Goal: move the ERes2NetV2 backbone training from the local RTX 4090 machine to `remote`,
where the challenge dataset is mounted in the same relative `datasets/` layout and the
Docker container exposes two NVIDIA H100 PCIe GPUs.

Remote paths:

- Host repository path: `<remote-repo>`
- Container repository path: `<repo-root>`
- Container: `container`

Preparation checks:

- Repository copied to the remote host while excluding local `.venv`, caches,
  `datasets/`, and large transient artifacts.
- Existing remote `datasets/` directory preserved.
- Required training/evaluation artifacts copied:
  - `artifacts/manifests/participants_fixed/`
  - `artifacts/baseline_fixed_participants/train_split.csv`
  - `artifacts/baseline_fixed_participants/val_split.csv`
  - `artifacts/eda/baseline_fixed_dense_shifted_v2_honest/`
  - `artifacts/eda/participants_audio6/file_stats.parquet`
- Container environment synced with:

```bash
uv sync --dev --group train
```

Validation:

- `scripts/validate_manifests.py --manifests-root artifacts/manifests/participants_fixed --strict`
  passed inside the container: `673277` valid rows, `2` manifests, `0` invalid rows.
- Training imports passed inside the container for `run_speaker_baseline` and
  `ERes2NetV2Encoder`.

Remote launch:

- Experiment id: `eres2netv2_h100_b64_20260411_200341`
- PID file: `artifacts/logs/eres2netv2_h100_b64_20260411_200341.pid`
- Log: `artifacts/logs/eres2netv2_h100_b64_20260411_200341.log`
- Latest run pointer: `artifacts/logs/latest_eres2netv2_h100_b64.txt`

```bash
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
uv run --group train python scripts/run_baseline.py \
  --model eres2netv2 \
  --config configs/training/eres2netv2-participants-candidate.toml \
  --device cuda
```

Decision: use the default H100 batch settings from the config (`batch_size=64`,
`eval_batch_size=64`) for the first remote launch because H100 has enough memory and
the local OOM was specific to the 24 GiB RTX 4090.

Status after launch: running inside `container` on GPU0. Initial remote check showed process
PID `248490`, H100 memory around `35 GiB`, and GPU utilization around `93%`; no startup
OOM observed.

Follow-up correction:

- The first remote launch was stopped intentionally because the training loop did not emit
  per-epoch/per-step progress to stdout, leaving the log empty during long epochs.
- Added stdout progress logging in `run_classification_batches`: epoch start, first batch,
  periodic batch progress, loss, accuracy, examples/sec, and elapsed seconds.
- Because `batch_size=64` used only about `35 GiB` on H100, the next ERes2NetV2 launch
  should use `training.batch_size=128` and `training.eval_batch_size=128` to improve
  throughput and make better use of the 80 GiB card.

Parallel GPU plan:

- GPU0: `ERes2NetV2` participants candidate, H100 batch/eval batch `128`.
- GPU1: `CAM++` participants candidate, H100 batch/eval batch `128` if it fits; fall back
  to `96` or `64` only on OOM.
- These are independent backbone hypotheses. The current code path is not DDP-enabled, so
  using both cards as separate jobs is safer and gives faster hypothesis coverage than
  forcing multi-GPU training into the existing single-process pipeline.

Active H100 launches after logging fix:

- `eres2netv2_h100_b128_20260411_200735`
  - GPU: `0`
  - Overrides: `training.batch_size=128`, `training.eval_batch_size=128`
  - Log: `artifacts/logs/eres2netv2_h100_b128_20260411_200735.log`
  - Initial status: alive, epoch `1/10`, `5155` batches/epoch, GPU memory about
    `69-70 GiB`.
- `campp_h100_b128_20260411_200735`
  - GPU: `1`
  - Overrides: `training.batch_size=128`, `training.eval_batch_size=128`
  - Status: stopped intentionally after startup because it used only about `9 GiB`; this
    did not make good use of the H100 card.
- `campp_h100_b1024_20260411_200846`
  - GPU: `1`
  - Overrides: `training.batch_size=1024`, `training.eval_batch_size=1024`
  - Log: `artifacts/logs/campp_h100_b1024_20260411_200846.log`
  - Initial status: alive, epoch `1/10`, `645` batches/epoch, GPU memory about
    `62-63 GiB`.

Decision: keep `ERes2NetV2 batch128` and `CAM++ batch1024` running as the current
parallel backbone training hypotheses. If a run OOMs during training or eval, restart
only that run with the next lower batch (`ERes2NetV2`: `112` or `96`; `CAM++`: `768`
or `512`).
