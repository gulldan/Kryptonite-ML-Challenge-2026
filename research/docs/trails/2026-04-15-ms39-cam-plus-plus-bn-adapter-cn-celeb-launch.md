# 2026-04-15 — MS39 CAM++ BN-Adapter CN-Celeb Launch

Hypothesis:

- After MS32, full encoder fine-tuning on new target-like signals has been fragile: late
  pseudo and clean stages improved local graph diagnostics while hurting hidden LB.
- The strong CAM++ base should be preserved. A parameter-efficient adaptation branch should
  freeze the encoder body and update only BatchNorm affine parameters plus BatchNorm running
  statistics, using participant + CN-Celeb supervised data for a short 1-2 epoch adaptation.
- Two starts are worth testing: MS31 as the clean strong supervised branch, and MS32 as the
  stronger filtered-pseudo branch. If the BN-only path helps, public C4 should stay close to
  the safe MS32/MS41 neighbourhood geometry instead of showing the large hidden-LB regressions
  observed in MS34-MS36.

Repository changes:

- Added `src/kryptonite/training/trainable_scope.py`.
- Extended `scripts/run_campp_finetune.py` with
  `--encoder-trainable-scope batchnorm-affine`.
- Added configs:
  `configs/training/campp-ms39-ms31-bn-adapter-cnceleb.toml` and
  `configs/training/campp-ms39b-ms32-bn-adapter-cnceleb.toml`.
- Added `scripts/run_ms39_bn_adapter_cnceleb.py` to wait for CN-Celeb extraction, build the
  mixed manifest, run both branches sequentially on one GPU, run the standard official CAM++
  public C4 tail, copy short submission files, and write a summary JSON.
- Local checks before sync:
  `uv run pytest tests/unit/test_trainable_scope.py`,
  `uv run ruff check src/kryptonite/training/trainable_scope.py scripts/run_campp_finetune.py scripts/run_ms39_bn_adapter_cnceleb.py tests/unit/test_trainable_scope.py`,
  `uvx ty check src/kryptonite/training/trainable_scope.py tests/unit/test_trainable_scope.py`.
- Full `ruff check .` and full `uvx ty check` were not clean before this change: they still
  report unrelated legacy issues in `baseline/`, `serve/`, and older tests.

Remote launch:

- Combined watcher id: `MS39_bn_adapter_cnceleb_20260415T0630Z`.
- Branch A id: `MS39_campp_ms31_bn_adapter_cnceleb_20260415T0630Z`.
- Branch B id: `MS39b_campp_ms32_bn_adapter_cnceleb_20260415T0630Z`.
- Execution environment: `<redacted>`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=1`.
- PID: `511026`.
- Log: `artifacts/logs/MS39_bn_adapter_cnceleb_20260415T0630Z.log`.
- PID file: `artifacts/logs/MS39_bn_adapter_cnceleb_20260415T0630Z.pid`.
- Latest pointer: `artifacts/logs/latest_MS39_bn_adapter_cnceleb.txt`.
- Report JSON:
  `artifacts/reports/ms39/MS39_bn_adapter_cnceleb_20260415T0630Z_summary.json`.
- MS37 full-finetune watcher stopped before launch: PIDs `499912` and `499932`, both were
  still only waiting for CN-Celeb extraction.
- CN-Celeb downloader remains active: PIDs `481739` and `481742`; archive size at MS39
  launch was `13.84 GiB`; `datasets/CN-Celeb_flac` was not present yet.

Branch A configuration:

- Config: `configs/training/campp-ms39-ms31-bn-adapter-cnceleb.toml`.
- Init checkpoint:
  `artifacts/baselines/campp-ms1-official-participants-voxblink2-augment/20260413T203245Z-b87036ccb3db/campp_encoder.pt`.
- Train manifest, after the watcher builds it:
  `artifacts/manifests/cnceleb_v2_ms39/cnceleb_v2_ms39_mixed_train_manifest.jsonl`.
- Dev manifest: `artifacts/manifests/participants_fixed/dev_manifest.jsonl`.
- Encoder scope: `batchnorm-affine`; all non-BN encoder parameters frozen, BN affine
  trainable, BN running stats updated through normal train mode.
- Objective: cosine proxy classifier, ArcMargin scale `32`, margin `0.2`,
  `subcenters_per_class=1`.
- Optimizer/scheduler: AdamW, LR `5e-5`, min LR `5e-6`, weight decay `0`, cosine schedule,
  no warmup, grad clip `3.0`.
- Precision/batch/epochs: bf16, batch `256`, max epochs `2`.
- Domain mix: domain-balanced loader with external share `0.30` and source prefixes
  `["cnceleb", "ffsvc"]`; current materialized data is expected to be CN-Celeb only.
- Crop/frontend: official CAM++ frontend, no VAD, fixed 6s train crops, 6s eval chunks,
  segment-mean public tail.

Branch B configuration:

- Config: `configs/training/campp-ms39b-ms32-bn-adapter-cnceleb.toml`.
- Init checkpoint:
  `artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt`.
- Same mixed train/dev manifests and encoder scope as Branch A.
- Optimizer/scheduler: AdamW, LR `4e-5`, min LR `4e-6`, weight decay `0`, cosine schedule,
  no warmup, grad clip `3.0`.
- Augmentation is slightly lighter than Branch A because the MS32 start is the branch most at
  risk of losing the useful filtered-pseudo geometry.

Launch command:

```bash
cd <repo-root>
RUN_GROUP=MS39_bn_adapter_cnceleb_20260415T0630Z
MS31_RUN=MS39_campp_ms31_bn_adapter_cnceleb_20260415T0630Z
MS32_RUN=MS39b_campp_ms32_bn_adapter_cnceleb_20260415T0630Z
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 uv run --group train \
  python scripts/run_ms39_bn_adapter_cnceleb.py \
    --ms31-run-id "$MS31_RUN" \
    --ms32-run-id "$MS32_RUN" \
    --ms31-tail-output-dir artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms39_campp_ms31_bn_adapter_cnceleb_20260415T0630Z \
    --ms32-tail-output-dir artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms39b_campp_ms32_bn_adapter_cnceleb_20260415T0630Z \
    --report-json artifacts/reports/ms39/${RUN_GROUP}_summary.json
```

Status at launch:

- Log first line:
  `[ms39] start 2026-04-15T06:31:38.741807+00:00 gpu=1 encoder_scope=batchnorm-affine`.
- Current state:
  `[ms39] waiting for datasets/CN-Celeb_flac; archive_size_gib=13.84`.
- GPU1 was idle (`0 MiB`, `0%`) while the watcher waited. GPU0 remained occupied by H9.

Decision:

- Pending CN-Celeb completion, training, public C4 validation, and public LB submission.
- If local C4 improves while public LB regresses like MS34-MS36, record that BN-only does not
  solve the hidden-transfer gap for this external data. If MS39b preserves MS32/MS41-like
  overlap and improves public, prefer it as the robust private-LB candidate.
