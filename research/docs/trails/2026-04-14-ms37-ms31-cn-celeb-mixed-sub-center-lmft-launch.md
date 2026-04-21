# 2026-04-14 — MS37 MS31 CN-Celeb Mixed Sub-Center LMFT Launch

Hypothesis:

- The late MS32-derived pseudo-label branches improved local public-graph diagnostics but
  repeatedly hurt hidden/public LB transfer. Return to the cleaner strong branch `MS31` and
  add real target-like external speaker supervision before any further public pseudo loop.
- CN-Celeb v2 is the available external target-like speaker corpus. Restricted FFSVC audio
  is still unavailable because `ZENODO_ACCESS_TOKEN` is not set; FFSVC remains a prepared
  follow-up path, not an audio source in this launch.
- Stage A should adapt the MS31 encoder with participant + CN-Celeb supervised labels while
  preventing CN-Celeb from dominating each batch. Stage B should be a distinct LMFT pass
  with longer crops, lower LR, larger margin, and almost no augmentation. The public tail
  should remain the MS32 C4 tail to isolate encoder/training effects.

Code/config changes:

- Added sub-center support to `CosineClassifier` through
  `objective.subcenters_per_class`; checkpointed classifiers now have
  `num_classes * subcenters_per_class` centers but expose max-pooled class logits.
- Added optional Inter-TopK impostor penalty through `objective.inter_topk_*`.
- Added optional domain-balanced speaker sampling through
  `training.domain_balance_enabled`, `training.domain_balance_external_share`, and
  `training.domain_balance_external_source_prefixes`.
- Added `--init-classifier-from-checkpoint` to `scripts/run_campp_finetune.py` so the LMFT
  stage restores the Stage A classifier and speaker index instead of starting from a random
  head.
- Added configs:
  `configs/training/campp-ms37-cnceleb-mixed-subcenter-lowlr.toml` and
  `configs/training/campp-ms37-cnceleb-mixed-subcenter-lmft.toml`.
- Added orchestrator:
  `scripts/run_ms37_cnceleb_subcenter_lmft.py`.

Local verification before remote launch:

```bash
uv run ruff check \
  src/kryptonite/models/campp/losses.py \
  src/kryptonite/training/baseline_config.py \
  src/kryptonite/training/baseline_pipeline.py \
  src/kryptonite/training/optimization_runtime.py \
  src/kryptonite/config.py \
  src/kryptonite/training/production_dataloader.py \
  scripts/run_campp_finetune.py \
  scripts/run_ms37_cnceleb_subcenter_lmft.py \
  tests/unit/test_production_dataloader.py

uvx ty check \
  src/kryptonite/models/campp/losses.py \
  src/kryptonite/training/baseline_config.py \
  src/kryptonite/training/baseline_pipeline.py \
  src/kryptonite/training/optimization_runtime.py \
  src/kryptonite/config.py \
  src/kryptonite/training/production_dataloader.py \
  scripts/run_campp_finetune.py \
  scripts/run_ms37_cnceleb_subcenter_lmft.py

uv run pytest \
  tests/unit/test_production_dataloader.py \
  tests/unit/test_eres2netv2_baseline.py::test_eres2netv2_encoder_produces_embeddings_and_margin_loss \
  -q
```

Result: all checks passed (`5` targeted tests).

Remote sync and launch:

- Normal rsync code sync was run to `remote` before launch, excluding `.venv/`,
  `.cache/`, `datasets/`, `artifacts/`, `.pytest_cache/`, and `.ruff_cache/`.
- Remote config smoke and remote `ruff check` passed inside `container`.
- Superseded old CN-Celeb watcher
  `MS33_campp_ms31_cnceleb_mixed_lowlr_public_c4_20260414T1545Z`, PID `482582`, was
  stopped so it cannot take GPU1 when CN-Celeb extraction appears.

Run id:

- `MS37_campp_ms31_cnceleb_mixed_subcenter_lmft_20260414T2134Z`

Remote execution:

- Execution environment: `<redacted>`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=1`.
- Launcher PID: `499912`.
- Log:
  `artifacts/logs/MS37_campp_ms31_cnceleb_mixed_subcenter_lmft_20260414T2134Z.log`.
- PID file:
  `artifacts/logs/MS37_campp_ms31_cnceleb_mixed_subcenter_lmft_20260414T2134Z.pid`.
- Latest pointer:
  `artifacts/logs/latest_MS37_campp_ms31_cnceleb_mixed_subcenter_lmft.txt`.
- Summary report target:
  `artifacts/reports/ms37/MS37_campp_ms31_cnceleb_mixed_subcenter_lmft_20260414T2134Z_summary.json`.

Launch command:

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 uv run --group train \
  python scripts/run_ms37_cnceleb_subcenter_lmft.py \
  --run-id MS37_campp_ms31_cnceleb_mixed_subcenter_lmft_20260414T2134Z \
  --tail-output-dir artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms37_cnceleb_subcenter_lmft_20260414T2134Z \
  --report-json artifacts/reports/ms37/MS37_campp_ms31_cnceleb_mixed_subcenter_lmft_20260414T2134Z_summary.json
```

Stage A planned parameters:

- Config: `configs/training/campp-ms37-cnceleb-mixed-subcenter-lowlr.toml`.
- Init checkpoint:
  `artifacts/baselines/campp-ms1-official-participants-voxblink2-augment/20260413T203245Z-b87036ccb3db/campp_encoder.pt`.
- Train manifest built after CN-Celeb extraction:
  `artifacts/manifests/cnceleb_v2_ms37/cnceleb_v2_ms37_mixed_train_manifest.jsonl`.
- Dataset/split: participant fixed train plus CN-Celeb v2 OpenSLR 82 external train split;
  dev is `artifacts/manifests/participants_fixed/dev_manifest.jsonl`, capped to `1024`
  rows.
- Sampler: speaker-balanced with domain-balanced external share `0.30`;
  external prefixes `["cnceleb", "ffsvc"]`.
- Precision/batch: bf16, train batch `256`, eval batch `256`, one crop per row.
- Crop/preprocessing: official CAM++ frontend, fixed 6s train crops, no VAD, 6s mean-pooled
  eval chunks.
- Augmentation: moderate MS31-like schedule, max two augmentations per sample, weak speed
  perturbation, noise/reverb/distance/codec/silence enabled.
- Objective: sub-center ArcMargin, `subcenters_per_class=3`, scale `32`, margin `0.2`,
  Inter-TopK penalty weight `0.02`, top-k `5`.
- Optimizer/scheduler: AdamW, LR `2e-5`, min LR `2e-6`, weight decay `5e-5`, cosine,
  warmup `1`, max epochs `2`, early stopping on train loss with patience `1`, restore best.

Stage B LMFT planned parameters:

- Config: `configs/training/campp-ms37-cnceleb-mixed-subcenter-lmft.toml`.
- Init checkpoint: Stage A checkpoint discovered by the orchestrator.
- Classifier init: restored from Stage A via `--init-classifier-from-checkpoint`.
- Train/dev manifests: same as Stage A.
- Precision/batch: bf16, train batch `128`, eval batch `128`.
- Crop/preprocessing: fixed 10s train crops, no VAD, no scheduled augmentation, no silence
  augmentation.
- Objective: sub-center ArcMargin, `subcenters_per_class=3`, scale `32`, margin `0.4`,
  Inter-TopK penalty weight `0.02`, top-k `5`.
- Optimizer/scheduler: AdamW, LR `1e-5`, min LR `1e-6`, weight decay `5e-5`, cosine,
  warmup `0`, max epochs `1`.

Post-LMFT public tail:

- Runner: `scripts/run_official_campp_tail.py`.
- Tail settings: same MS32 C4 public tail, torch backend, official CAM++ segment-mean
  frontend, public packed frontend cache when available, batch `512`, search batch `2048`,
  top-cache `200`, `eval_chunk_seconds=6.0`, `segment_count=3`,
  `long_file_threshold_seconds=6.0`.
- Public output root:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms37_cnceleb_subcenter_lmft_20260414T2134Z`.

Initial status:

- MS37 watcher started at `2026-04-14T21:35:26Z`.
- It is waiting for `datasets/CN-Celeb_flac`.
- Active CN-Celeb downloader:
  `EXTDATA_cnceleb_resume_until_md5_20260414T1527Z`, PIDs `481721` / `481739` /
  `481742`; archive progress at launch was about `7.69 GiB`, `37%`, with the wget log
  estimating about `18h57m` remaining.
- GPU0 remains occupied by H9; GPU1 is idle while MS37 waits for the dataset directory.
