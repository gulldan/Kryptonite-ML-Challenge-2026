# 2026-04-13 — MS31 Official CAM++ VoxBlink2-Like Augmentation Launch

Hypothesis:

- VoxBlink2 reports on-the-fly data augmentation and speed perturbation during pretraining,
  followed by lower-LR large-margin fine-tuning without augmentation. This run intentionally
  tests the risky part for this challenge: mixing a VoxBlink2-like augmentation profile into
  the low-LR participant-train adaptation of the exact MS1 official CAM++ branch.
- Compared with MS30, this run keeps the official 3D-Speaker/ModelScope frontend,
  `kaldi.fbank(dither=0.0)`, utterance mean normalization, fixed 6s train crops, and public
  `segment_mean` 3x6s inference. The only strategic change is a stronger on-the-fly
  augmentation distribution.
- If this damages MS1 ranking geometry, reject this path and keep augmentation for a
  separate pretraining/pseudo-label stage. If it improves graph quality, use it as evidence
  for augmentation-aware public pseudo-label self-training.

Remote launch:

- Run id: `MS31_campp_ms1_official_voxblink2_aug_public_c4_20260413T2029Z`.
- Execution environment: `<redacted>`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=1`.
- Training config:
  `configs/training/campp-ms1-official-participants-voxblink2-augment.toml`.
- Init checkpoint:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/converted/modelscope_campplus_voxceleb_encoder.pt`.
- Train manifest: `artifacts/manifests/participants_fixed/train_manifest.jsonl`.
- Dev manifest: `artifacts/manifests/participants_fixed/dev_manifest.jsonl`.
- Model: CAM++ `512d`, ModelScope VoxCeleb pretrained initialization.
- Frontend: `features.frontend="official_campp"`, implemented as
  `torchaudio.compliance.kaldi.fbank(..., dither=0.0)` followed by utterance mean
  normalization.
- Crop/preprocess policy: no VAD, fixed 6s training crops with repeat padding for short
  utterances; dev export uses 6s chunks and mean pooling.
- Augmentation policy: warmup `1`, ramp `3`, max `3` augmentations/sample; clean ratio
  ramps `0.50 -> 0.08`; medium/heavy ratios ramp to `0.42/0.20`; family weights noise
  `1.15`, reverb `1.05`, distance `0.80`, codec `1.20`, silence `0.65`, speed `1.30`.
- Available augmentation banks on remote before launch: MUSAN direct noise `2016`,
  RIRS direct reverb `60417`, distance `3`, codec `7`, silence `3`, speed `4`;
  missing families none. Smoke report log:
  `artifacts/logs/MS31_campp_ms1_official_voxblink2_aug_public_c4_20260413T2029Z_augmentation_smoke.log`.
- Optimizer/scheduler: AdamW, LR `1e-4`, min LR `1e-5`, weight decay `5e-5`, cosine
  schedule, warmup `1` epoch, grad clip `5.0`.
- Precision/batch: bf16, batch `256`, eval batch `256`, max epochs `8`, early stopping on
  train loss after min epoch `4`, stop train accuracy threshold `0.9975`.
- Output root: `artifacts/baselines/campp-ms1-official-participants-voxblink2-augment/`.
- Tracking root: `artifacts/tracking/`.
- Public output root:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms31_voxblink2_aug_20260413T2029Z/`.
- Remote log:
  `artifacts/logs/MS31_campp_ms1_official_voxblink2_aug_public_c4_20260413T2029Z.log`.
- Latest pointer:
  `artifacts/logs/latest_MS31_campp_ms1_official_voxblink2_aug_public_c4.txt`.

Planned launch command:

```bash
cd <repo-root>
RUN_ID=MS31_campp_ms1_official_voxblink2_aug_public_c4_20260413T2029Z
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 uv run --group train \
  python scripts/run_campp_finetune.py \
  --config configs/training/campp-ms1-official-participants-voxblink2-augment.toml \
  --init-checkpoint artifacts/backbone_public/modelscope_campplus_voxceleb_default/converted/modelscope_campplus_voxceleb_encoder.pt \
  --device cuda \
  --output json

CHECKPOINT=$(python - <<'PY'
from pathlib import Path
root = Path("artifacts/baselines/campp-ms1-official-participants-voxblink2-augment")
runs = sorted((path for path in root.iterdir() if (path / "campp_encoder.pt").is_file()), key=lambda path: path.stat().st_mtime)
print(runs[-1] / "campp_encoder.pt")
PY
)

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 uv run --group train \
  python scripts/run_official_campp_tail.py \
  --checkpoint-path "$CHECKPOINT" \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --data-root 'datasets/Для участников' \
  --output-dir artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms31_voxblink2_aug_20260413T2029Z \
  --experiment-id "$RUN_ID" \
  --encoder-backend torch \
  --device cuda \
  --search-device cuda \
  --batch-size 512 \
  --search-batch-size 2048 \
  --top-cache-k 200 \
  --mode segment_mean \
  --eval-chunk-seconds 6.0 \
  --segment-count 3 \
  --long-file-threshold-seconds 6.0
```

Status:

- Training completed on remote GPU1. Final epoch metrics: train loss `1.343605`,
  train accuracy `0.955046`.
- Checkpoint:
  `artifacts/baselines/campp-ms1-official-participants-voxblink2-augment/20260413T203245Z-b87036ccb3db/campp_encoder.pt`.
- The original combined detached job did not reach the public tail. It completed training
  and wrote the checkpoint/dev embeddings, then stopped during the generic dev scoring path
  after creating a very large `dev_trials.jsonl`. This is diagnostic for the wrapper, not a
  model failure.
- Standalone public tail was rerun from the checkpoint using the packed official frontend
  cache `artifacts/cache/campp-official-public-ms1-v1-pack`.
- Tail log:
  `artifacts/logs/MS31_campp_ms1_official_voxblink2_aug_public_c4_20260413T2029Z_tail_submit_20260414T0527Z.log`.
- Public C4 tail metrics: embedding `160.909s`, exact top-k search `0.755s`,
  C4 rerank `9.639s`, `c4_top10_mean_score_mean=0.6206907`, label_used_share
  `0.8168`, Gini@10 `0.337978`, max in-degree `91`.
- Validator: passed, `134697/134697` rows, `k=10`, `error_count=0`.
- Submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms31_voxblink2_aug_20260413T2029Z/submission.csv`
  (short copy of
  `submission_MS31_campp_ms1_official_voxblink2_aug_public_c4_20260413T2029Z_c4.csv`).
- SHA-256:
  `cc00f3ca5c1b6de734b1a819c2c53fbacaf29a7afe4de17f76154591f707cb70`.
- Public LB score: `0.7018`.
- Result: new public best, `+0.1323` absolute vs MS1 `0.5695` and `+0.6239`
  absolute vs organizer baseline `0.0779`. This confirms the VoxBlink2-like
  augmentation mix helped the official CAM++ adaptation branch.

Current public best:

- User-reported external best:
  `MS31_campp_ms1_official_voxblink2_aug_public_c4_20260413T2029Z`, public LB
  `0.7018`.
- Artifact:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms31_voxblink2_aug_20260413T2029Z/submission.csv`

- Fast public probe candidate:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/official_graph_20260412T_candidate/submission_C1_b8_mutual20_component.csv`
  (`MS6_official_campp_c1_component_candidate`, validator passed, public score pending).

- Fast corrected remote CAM++ candidate:
  `artifacts/backbone_public/campp/submission_MS7_new_code_campp_from_scratch_official_frontend_c4.csv`
  (`MS7_campp_from_scratch_official_frontend_recalc`, validator passed, public LB `0.2597`; rejected as a replacement for MS1).

- Best repo-local scored artifact:
  `P3_eres2netv2_g6_pseudo_ft_public_c4`, public LB `0.2861`
- Artifact:
  `artifacts/backbone_public/eres2netv2_g6_pseudo_ft/20260412T100738Z-6b686847f5d8/submission_P3_eres2netv2_g6_pseudo_ft_public_c4.csv`
- Latest orthogonal candidate: `E1_wavlm_domain_ft_public_c4`, public LB `0.2833`
  with artifact
  `artifacts/backbone_public/hf_xvector/wavlm_e1_domain_ft_public_c4_20260412T130254Z/submission_E1_wavlm_domain_ft_public_c4.csv`.
