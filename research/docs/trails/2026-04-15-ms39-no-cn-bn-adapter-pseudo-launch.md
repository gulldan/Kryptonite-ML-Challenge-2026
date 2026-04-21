# 2026-04-15 — MS39 No-CN BN-Adapter Pseudo Launch

Reason for change:

- The CN-Celeb branch was blocked by download time. At 2026-04-15T06:35Z the archive was
  still only about `14 GiB`, the downloader reported `66%`, and ETA was about `9h30m`.
- User requested a no-CN path because there was not enough time to wait.
- The available target-like signal is the already materialized MS32 filtered public pseudo
  mixed manifest. This keeps the spirit of PEFT domain adaptation but avoids the external
  dataset dependency.

Repository changes:

- Added no-CN configs:
  `configs/training/campp-ms39-ms31-bn-adapter-pseudo.toml` and
  `configs/training/campp-ms39b-ms32-bn-adapter-pseudo-refresh.toml`.
- Both configs use `artifacts/manifests/pseudo_ms31/ms31_filtered_mixed_train_manifest.jsonl`,
  official CAM++ frontend, fixed 6s crops, one epoch, AdamW, weight decay `0`, and
  `--encoder-trainable-scope batchnorm-affine`.
- Branch A starts from MS31 and necessarily uses a fresh classifier head because the pseudo
  manifest speaker index differs from the MS31 participant-only checkpoint.
- Branch B starts from MS32 and uses `--init-classifier-from-checkpoint`. The remote check
  confirmed the pseudo manifest and MS32 checkpoint both have `14021` speakers and matching
  `speaker_to_index`.

Stopped jobs:

- Stopped CN watcher PIDs `511026` and `511029`.
- Stopped CN downloader PIDs `481739` and `481742`.
- Earlier MS37 full-finetune watcher had already been stopped before MS39 CN launch.

Remote launch:

- Run group: `MS39_no_cn_bn_adapter_pseudo_20260415T0639Z`.
- Branch A id: `MS39_campp_ms31_bn_adapter_pseudo_20260415T0639Z`.
- Branch B id: `MS39b_campp_ms32_bn_adapter_pseudo_refresh_20260415T0639Z`.
- Execution environment: `<redacted>`.
- GPU assignment: physical GPU1 through `CUDA_VISIBLE_DEVICES=1`.
- PID: `511411`.
- Log: `artifacts/logs/MS39_no_cn_bn_adapter_pseudo_20260415T0639Z.log`.
- PID file: `artifacts/logs/MS39_no_cn_bn_adapter_pseudo_20260415T0639Z.pid`.
- Latest pointer: `artifacts/logs/latest_MS39_no_cn_bn_adapter_pseudo.txt`.
- Report JSON:
  `artifacts/reports/ms39/MS39_no_cn_bn_adapter_pseudo_20260415T0639Z_summary.json`.

Branch A command:

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 uv run --group train \
  python scripts/run_campp_finetune.py \
    --config configs/training/campp-ms39-ms31-bn-adapter-pseudo.toml \
    --init-checkpoint artifacts/baselines/campp-ms1-official-participants-voxblink2-augment/20260413T203245Z-b87036ccb3db/campp_encoder.pt \
    --device cuda \
    --encoder-trainable-scope batchnorm-affine \
    --output text
```

Branch B command:

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 uv run --group train \
  python scripts/run_campp_finetune.py \
    --config configs/training/campp-ms39b-ms32-bn-adapter-pseudo-refresh.toml \
    --init-checkpoint artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt \
    --init-classifier-from-checkpoint \
    --device cuda \
    --encoder-trainable-scope batchnorm-affine \
    --output text
```

Status:

- Branch A started training at 2026-04-15T06:40Z.
- Encoder trainable scope:
  `81792/7176224` trainable encoder params, `1.1398%`, `122` BatchNorm modules.
- Training has `2986` batches for the one epoch.
- Branch B launched after Branch A completed and reused the MS32-compatible
  `14021`-speaker classifier head.
- Run group completed at `2026-04-15T07:48:46Z`.

Results:

- Branch A `MS39_campp_ms31_bn_adapter_pseudo_20260415T0639Z` completed on remote GPU1.
  Local C4 validator passed; `top10_mean_score_mean=0.6598`, `label_used_share=0.8406`,
  Gini@10 `0.3323`, max in-degree `81`. Submission:
  `artifacts/submissions/MS39_campp_ms31_bn_adapter_pseudo_20260415T0639Z_submission.csv`
  with SHA-256
  `d22a53c83115e143da13200803feddab83880fa3ec60d4ca308c9f5e6d627848`.
- User-submitted public leaderboard score for Branch A on 2026-04-15:
  `0.7031`.
- Branch B `MS39b_campp_ms32_bn_adapter_pseudo_refresh_20260415T0639Z` also completed in
  the same GPU1 run group. Local C4 validator passed; `top10_mean_score_mean=0.6536`,
  `label_used_share=0.8928`, Gini@10 `0.3330`, max in-degree `57`. Submission:
  `artifacts/submissions/MS39b_campp_ms32_bn_adapter_pseudo_refresh_20260415T0639Z_submission.csv`
  with SHA-256
  `c671933363ad4f15277809386a9641ec508e72742d81a43816a8922b440940a9`.
- Branch B was not submitted publicly because its local proxy finished below Branch A.

Decision:

- Reject Branch A as a replacement for `MS32`/`MS41`: public LB `0.7031` is only
  `+0.0013` vs `MS31`, but `-0.0348` vs `MS32` and `-0.0442` vs `MS41`.
- Do not spend a public submission on Branch B under the current budget; it completed
  with a weaker local proxy than Branch A and does not justify an extra leaderboard probe.
