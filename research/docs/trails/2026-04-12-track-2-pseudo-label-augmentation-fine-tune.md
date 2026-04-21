# 2026-04-12 - Track 2 Pseudo-Label Augmentation Fine-Tune

Experiment id: `P4_eres2netv2_g6_pseudo_track2_aug_e70`

Hypothesis:

- The current public best is `P3_eres2netv2_g6_pseudo_ft_public_c4` with public LB
  `0.2861`, so pseudo-label self-training is confirmed useful. The next orthogonal
  hypothesis is to keep the P3 pseudo-label branch and reduce the measured public-domain
  gap with a production training augmentation package.
- A1 must-have augmentation: real noise/music/babble-style noise bank, RIR convolution,
  and speed perturbation.
- A2 test-matching augmentation: channel/codec-like band limiting and quantization,
  random EQ, far-field attenuation, trailing silence, inserted pauses, and random
  VAD-drop. This targets the EDA finding that public test has less high-frequency energy
  and more pauses.

Code/config changes:

- Added production scheduler/runtime wiring in `ManifestSpeakerDataset` and
  `build_production_train_dataloader`, so scheduled waveform augmentations are applied
  before random crop and fbank extraction.
- Added direct raw MUSAN and RIRS_NOISES fallbacks to the augmentation runtime. Training
  only registers noise/RIR candidates whose audio files exist, so remote downloads are now
  used directly instead of relying on prebuilt artifact banks.
- Fixed `scripts/download_datasets.py` so downloads executed with `cwd=datasets` write
  archives as `musan.tar.gz` / `rirs_noises.zip`, not `datasets/<archive>` from inside the
  `datasets` directory.
- Switched MUSAN primary download to the full Hugging Face LFS zip mirror
  `thusinh1969/musan` after OpenSLR/trmal stalled at only tens of kilobytes. Switched
  RIRS_NOISES primary download to the Hugging Face zip mirror `EaseZh/rirs_noises`, with
  OpenSLR zip URLs retained as fallback.
- Updated RIRS_NOISES discovery to accept both `datasets/rirs_noises/RIRS_NOISES` and
  the actual extracted `datasets/RIRS_NOISES` layout; `scripts/download_datasets.py` now
  treats `datasets/RIRS_NOISES` as the downloaded directory.
- Optimized the production dataloader/scheduler for the large raw RIR catalog: sampler
  batches are yielded lazily instead of materializing a full epoch before batch 1, and
  augmentation candidate pools/cumulative weights are cached by family and severity.
- Added `configs/training/eres2netv2-g6-pseudo-track2-augment.toml`.
- The config explicitly opts into speed perturbation with
  `augmentation_scheduler.family_weights.speed=0.75`; older configs without this field
  keep speed disabled.

Training configuration:

- Config path: `configs/training/eres2netv2-g6-pseudo-track2-augment.toml`.
- Model family: ERes2NetV2, same architecture as P3, embedding size `192`.
- Initialization/provenance:
  `artifacts/baselines/eres2netv2-g6-pseudo-finetune/20260412T100738Z-6b686847f5d8/eres2netv2_encoder.pt`.
- Train manifest:
  `artifacts/manifests/pseudo_g6/g6_mixed_train_manifest.jsonl`.
- Dev manifest:
  `artifacts/manifests/participants_fixed/dev_manifest.jsonl`.
- Dataset/split version: original participant train plus filtered G6 public pseudo
  clusters; dev is `participants_fixed`.
- Seed: `42`.
- Batch size: `128`; eval batch size: `128`.
- Precision: `bf16`.
- Optimizer/scheduler: SGD momentum `0.9`, weight decay `0.00005`, cosine LR,
  initial LR `0.003`, min LR `0.000003`, warmup epochs `3`.
- Objective/loss: ArcMargin classifier, scale `32.0`, margin `0.2`, easy margin `false`.
- Crop/preprocessing: random train crop `2.0..6.0s`, one crop, eval chunks `6.0s`
  with `1.5s` overlap, VAD disabled.
- Augmentation policy: warmup `2` epochs, ramp `8` epochs, max `3` augmentations per
  sample, clean/light/medium/heavy ramps from `0.55/0.35/0.10/0.00` to
  `0.20/0.30/0.30/0.20`; family weights noise `1.20`, reverb `1.00`, distance `0.95`,
  codec `1.10`, silence `0.85`, speed `0.75`.
- Early stopping: `max_epochs=70` as an upper cap, `early_stopping_enabled=true`,
  monitor `train_loss`, `min_delta=0.0005`, patience `8`, min epochs `12`,
  restore best state `true`, train-accuracy hard stop `0.9975`.
- GPU/device assignment: `remote`, container `container`, `CUDA_VISIBLE_DEVICES=0`.
- Container/environment: `<repo-root>`, `uv sync --dev --group train`.
- Local validation design: final pipeline dev scoring after training. Per-epoch dev is
  not currently part of this production loop; early stopping uses persisted per-epoch
  training metrics, and the final checkpoint is scored on `participants_fixed` dev.

Pre-launch scheduler smoke:

- Local command:

```bash
uv run python - <<'PY'
from pathlib import Path
from kryptonite.training.eres2netv2 import load_eres2netv2_baseline_config
from kryptonite.training.augmentation_scheduler import build_augmentation_scheduler_report
cfg = load_eres2netv2_baseline_config(
    config_path=Path("configs/training/eres2netv2-g6-pseudo-track2-augment.toml")
)
report = build_augmentation_scheduler_report(
    project_root=Path("."),
    scheduler_config=cfg.project.augmentation_scheduler,
    silence_config=cfg.project.silence_augmentation,
    total_epochs=cfg.project.training.max_epochs,
    samples_per_epoch=256,
    seed=cfg.project.runtime.seed,
)
print(report.catalog.candidate_counts_by_family)
print(report.summary.missing_families)
PY
```

- Candidate counts after direct raw MUSAN/RIRS fallback: noise `2859`, reverb `60437`,
  distance `3`, codec `7`, silence `3`,
  speed `4`.
- Missing families: none.

Remote launch records:

- Remote run id: `eres2netv2_g6_pseudo_track2_aug_e70_20260412T144522Z`.
- Remote log path:
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T144522Z.log`.
- Result: failed before training. The download wrapper attempted to save to
  `datasets/musan.tar.gz` while running with `cwd=datasets`, so `wget` exited with
  `datasets/musan.tar.gz: No such file or directory`. No GPU training started.
- Remote run id: `eres2netv2_g6_pseudo_track2_aug_e70_20260412T145307Z`.
- Remote log path:
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T145307Z.log`.
- Result: failed before training for the same downloader path issue because the remote
  script had not yet received the fixed `scripts/download_datasets.py`. No GPU training
  started.
- Remote run id: `eres2netv2_g6_pseudo_track2_aug_e70_20260412T145511Z`.
- Remote log path:
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T145511Z.log`.
- Result: stopped before training. It used the fixed path handling, but the OpenSLR/trmal
  MUSAN transfer stalled at roughly `70K`; no GPU training started.
- Remote run id: `eres2netv2_g6_pseudo_track2_aug_e70_20260412T150004Z`.
- Remote log path:
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T150004Z.log`.
- Result: downloaded and extracted MUSAN and RIRS_NOISES successfully, then failed at the
  augmentation smoke before training because RIRS extracted to `datasets/RIRS_NOISES`
  while the first runtime lookup only checked `datasets/rirs_noises/RIRS_NOISES`.
- Remote run id: `eres2netv2_g6_pseudo_track2_aug_e70_20260412T151236Z`.
- Remote log path:
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T151236Z.log`.
- Result: stopped before batch 1 after the RIRS path fix. The train process entered epoch
  1 but spent the startup window materializing all `5814` batches and repeatedly filtering
  the large RIR catalog. This exposed a scheduler/dataloader performance bug; no useful
  training metrics were produced.
- Remote run id: `eres2netv2_g6_pseudo_track2_aug_e70_20260412T151705Z`.
- Remote log path:
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T151705Z.log`.
- Remote pid path:
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T151705Z.pid`.
- Latest pointer:
  `artifacts/logs/latest_eres2netv2_g6_pseudo_track2_aug_e70`.
- Output root:
  `artifacts/baselines/eres2netv2-g6-pseudo-track2-augment/<tracking-run-id>/`.

Launch command:

```bash
<remote-shell> <remote-host> 'docker exec -i container bash' <<'REMOTE'
set -euo pipefail
cd <repo-root>
RUN_ID=eres2netv2_g6_pseudo_track2_aug_e70_20260412T151705Z
mkdir -p artifacts/logs datasets
printf '%s\n' "$RUN_ID" > artifacts/logs/latest_eres2netv2_g6_pseudo_track2_aug_e70
cat > "/tmp/${RUN_ID}.sh" <<'JOB'
#!/usr/bin/env bash
set -euo pipefail
cd <repo-root>
RUN_ID=eres2netv2_g6_pseudo_track2_aug_e70_20260412T151705Z
uv sync --dev --group train
uv run python scripts/download_datasets.py --dataset musan
uv run python scripts/download_datasets.py --dataset rirs-noises
uv run python - <<'PY'
from pathlib import Path
from kryptonite.training.eres2netv2 import load_eres2netv2_baseline_config
from kryptonite.training.augmentation_runtime import TrainingAugmentationRuntime
cfg = load_eres2netv2_baseline_config(
    config_path=Path("configs/training/eres2netv2-g6-pseudo-track2-augment.toml")
)
runtime = TrainingAugmentationRuntime.from_project_config(
    project_root=Path("."),
    scheduler_config=cfg.project.augmentation_scheduler,
    silence_config=cfg.project.silence_augmentation,
    total_epochs=cfg.project.training.max_epochs,
)
counts = runtime.catalog.candidate_counts_by_family
print(counts, flush=True)
required = {"noise", "reverb", "codec", "silence", "speed"}
missing = sorted(required.difference(counts))
if missing:
    raise SystemExit(f"missing augmentation families: {missing}")
PY
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
uv run --group train python scripts/run_eres2netv2_finetune.py \
  --config configs/training/eres2netv2-g6-pseudo-track2-augment.toml \
  --init-checkpoint artifacts/baselines/eres2netv2-g6-pseudo-finetune/20260412T100738Z-6b686847f5d8/eres2netv2_encoder.pt \
  --device cuda \
  --output json
JOB
chmod +x "/tmp/${RUN_ID}.sh"
nohup "/tmp/${RUN_ID}.sh" > "artifacts/logs/${RUN_ID}.log" 2>&1 &
echo $! > "artifacts/logs/${RUN_ID}.pid"
REMOTE
```

Status:

- `eres2netv2_g6_pseudo_track2_aug_e70_20260412T151705Z` launched detached on remote
  GPU0. Downloads are complete and reused: `datasets/musan` and `datasets/RIRS_NOISES`.
  Remote smoke counts: noise `2016`, reverb `60417`, distance `3`, codec `7`,
  silence `3`, speed `4`; missing families none. As of `2026-04-12T15:18:04Z`, training
  reached `epoch=1/70 batch=1/5814`, batch-1 loss `17.890003`, accuracy `0.000000`,
  throughput `33.8` examples/s, and GPU0 was active at roughly `74905/81559 MiB` and
  `100%` utilization.
- Hourly monitor started with pid path
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T151705Z.hourly_monitor.pid`
  and monitor log
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T151705Z.hourly_monitor.log`.
  First monitor snapshot at `2026-04-12T17:10:08Z`: training was running, epoch 1
  completed with train loss `10.206978` and accuracy `0.416613`; epoch 2 had reached
  `batch=4640/5814` (`79.8%`), train loss `3.319962`, accuracy `0.878155`, throughput
  about `201.7` examples/s, GPU0 `76789/81559 MiB`, `100%` utilization.
- Final training outcome checked on `2026-04-13T03:53:47Z`: early stopping fired at
  epoch `12` with reason `patience_exhausted`; best epoch was `4`, best train loss
  `2.519065`, and `restore_best=true` restored the best checkpoint before writing.
  Final epoch 12 train loss was `2.965722` and train accuracy `0.873003`.
- Training artifacts:
  checkpoint
  `artifacts/baselines/eres2netv2-g6-pseudo-track2-augment/20260412T151747Z-b522b570a1b7/eres2netv2_encoder.pt`;
  training summary
  `artifacts/baselines/eres2netv2-g6-pseudo-track2-augment/20260412T151747Z-b522b570a1b7/training_summary.json`;
  dev embeddings
  `artifacts/baselines/eres2netv2-g6-pseudo-track2-augment/20260412T151747Z-b522b570a1b7/dev_embeddings.npz`.
- The training wrapper was manually stopped after checkpoint/summary/dev embeddings were
  written because the generic baseline pipeline moved into all-pairs dev trial generation:
  this config had `trials_manifest=""`, and `participants_fixed` dev has `13473` rows.
  That scorer path would materialize an impractically large in-memory trial list and is
  not the intended public-submission evaluation path for this branch. No `score_summary`
  or verification report was produced by this run.
- GPU0 was freed after stopping the post-train scorer tail; `nvidia-smi` showed
  `0 MiB` and `0%` utilization on both GPUs.

P4 public C4 tail launch:

- Purpose: create a leaderboard submission from the Track 2 augmentation checkpoint using
  the same public C4 tail as the current P3 best (`top-cache-k=200`, `3` crops,
  `6.0s`, no synthetic shift), so the LB comparison is direct.
- Run id: `p4_eres2netv2_track2_aug_public_c4_20260413T035400Z`.
- GPU: `CUDA_VISIBLE_DEVICES=0`.
- Log:
  `artifacts/logs/p4_eres2netv2_track2_aug_public_c4_20260413T035400Z.log`.
- PID file:
  `artifacts/logs/p4_eres2netv2_track2_aug_public_c4_20260413T035400Z.pid`.
- Checkpoint:
  `artifacts/baselines/eres2netv2-g6-pseudo-track2-augment/20260412T151747Z-b522b570a1b7/eres2netv2_encoder.pt`.
- Output dir:
  `artifacts/backbone_public/eres2netv2_track2_aug/20260412T151747Z-b522b570a1b7/`.
- Exact launch command:

```bash
cd <repo-root>
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
uv run --group train python scripts/run_torch_checkpoint_c4_tail.py \
  --model eres2netv2 \
  --checkpoint-path artifacts/baselines/eres2netv2-g6-pseudo-track2-augment/20260412T151747Z-b522b570a1b7/eres2netv2_encoder.pt \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/eres2netv2_track2_aug/20260412T151747Z-b522b570a1b7 \
  --experiment-id P4_eres2netv2_g6_pseudo_track2_aug_public_c4 \
  --shift-mode none \
  --device cuda \
  --search-device cuda \
  --precision bf16 \
  --batch-size 256 \
  --search-batch-size 4096 \
  --top-cache-k 200 \
  --crop-seconds 6.0 \
  --n-crops 3
```

- Initial monitor: embedding extraction started on `134697` public rows; GPU0 used
  about `21945/81559 MiB` with `100%` utilization.
