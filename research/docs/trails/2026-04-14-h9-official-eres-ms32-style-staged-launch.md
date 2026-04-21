# 2026-04-14 — H9 Official ERes MS32-Style Staged Launch

Hypothesis:

- The transferable part of MS32 is not extra epochs. It is stage-wise adaptation from a
  strong official/pretrained geometry, followed by conservative filtered public
  pseudo-label self-training.
- This run applies that structure to official 3D-Speaker ERes2Net-large:
  supervised H8 low-LR adaptation first, H8 public graph pseudo-pool second, then H9
  filtered pseudo self-training.
- This is intentionally separate from the from-scratch `P1/P3/P4` ERes2NetV2 lineage.
  It tests whether official ERes initialization and frontend alignment are the missing
  piece, not whether more epochs rescue the scratch branch.

Repository changes:

- Added `scripts/run_official_3dspeaker_eres2net_finetune.py`, a thin runner that loads
  `speakerlab.models.eres2net.ERes2Net` from `/tmp/3D-Speaker`, initializes from either a
  raw official state dict or this repository's training checkpoint payload, and delegates
  to `run_speaker_baseline()`.
- Updated `scripts/run_official_3dspeaker_eres2net_tail.py` so the public tail can load
  fine-tuned training checkpoints containing `model_state_dict`.
- Added configs:
  `configs/training/official-3dspeaker-eres2net-large-participants-lowlr.toml` and
  `configs/training/official-3dspeaker-eres2net-large-filtered-pseudo-lowlr.toml`.
- Local code state at launch: commit `8aa81b4` plus the uncommitted official ERes runner,
  tail loader patch, two configs, and this experiment-history entry.

Stage H8 supervised low-LR adaptation:

- Config:
  `configs/training/official-3dspeaker-eres2net-large-participants-lowlr.toml`.
- Init checkpoint:
  `artifacts/modelscope_cache/official_3dspeaker/speech_eres2net_large_sv_zh-cn_3dspeaker_16k/eres2net_large_model.ckpt`.
- Model family: official 3D-Speaker `ERes2Net`, `feat_dim=80`, `embedding_size=512`,
  `m_channels=64`, model id `iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k`.
- Train/dev manifests: `artifacts/manifests/participants_fixed/train_manifest.jsonl` and
  `artifacts/manifests/participants_fixed/dev_manifest.jsonl`; dev capped to `1024` rows
  to avoid the all-pairs scorer blowup.
- Seed/environment: seed `42`, remote container `container`, repo path
  `<repo-root>`, bf16 precision, batch `96`,
  gradient accumulation `1`, GPU0.
- Frontend/crop: 80-bin Kaldi fbank with utterance mean normalization via
  `features.frontend="official_campp"`, fixed 6s train crops, VAD disabled.
- Optimizer/scheduler: AdamW, LR `5e-5`, min LR `5e-6`, weight decay `5e-5`, cosine,
  warmup `1`, max epochs `4`, early stopping on train loss with min epochs `2`,
  patience `2`, restore best.
- Objective: ArcMargin scale `32.0`, margin `0.2`.
- Augmentation: conservative MS31-like public-shift schedule, max two augmentations per
  sample, clean probability `0.65 -> 0.35`, family weights noise `1.10`, reverb `0.85`,
  distance `0.90`, codec `1.05`, silence `1.00`, speed `0.35`.

Stage H8 public pseudo-pool:

- Public tail output:
  `artifacts/backbone_public/official_3dspeaker_eres2net_large_h8_lowlr_20260414T2113Z/`.
- Tail settings: official 3D-Speaker ERes runner, bf16, batch `128`, top-cache `200`,
  C4 label-propagation rerank.
- Pseudo-pool id: `H8_official_eres_clusterfirst_shared4_penalty020_top200_20260414T2113Z`.
- Cluster-first settings match MS32's pseudo pool shape: `edge_top=20`,
  `reciprocal_top=50`, `rank_top=200`, `iterations=8`, cluster sizes `[5,160]`,
  `cluster_min_candidates=3`, `shared_top=50`, `shared_min_count=4`,
  `split_edge_top=8`, `self_weight=0.0`, `label_size_penalty=0.20`.
- Filtered pseudo manifest output:
  `artifacts/manifests/pseudo_h8_official_eres/h8_official_eres_filtered_mixed_train_manifest.jsonl`,
  with pseudo cluster size filter `[8,80]`.

Stage H9 filtered pseudo self-training:

- Config:
  `configs/training/official-3dspeaker-eres2net-large-filtered-pseudo-lowlr.toml`.
- Init checkpoint: H8 checkpoint discovered after supervised training.
- Train manifest:
  `artifacts/manifests/pseudo_h8_official_eres/h8_official_eres_filtered_mixed_train_manifest.jsonl`.
- Precision/batch: bf16, train batch `96`, gradient accumulation `1`, eval batch `128`.
- Optimizer/scheduler: AdamW, LR `2e-5`, min LR `2e-6`, weight decay `5e-5`, cosine,
  warmup `1`, max epochs `4`, early stopping on train loss with min epochs `2`,
  patience `2`, restore best.
- Objective: hard ArcMargin over original speakers plus pseudo public cluster labels,
  scale `32.0`, margin `0.2`, pseudo loss weight `1.0`.
- Public tail output:
  `artifacts/backbone_public/official_3dspeaker_eres2net_large_h9_pseudo_20260414T2113Z/`.

Remote launch plan:

- Failed run id: `H9_official_eres_filtered_pseudo_public_c4_20260414T2057Z`.
- Failure result: H8 supervised train OOMed on GPU0 at batch `128` before completing the
  first epoch. GPU0 was freed by killing PID `498099`. No checkpoint, pseudo pool, or
  public submission was produced.
- Batch-64 retry run id: `H9_official_eres_filtered_pseudo_public_c4_20260414T2104Z`.
- Batch-64 result: manually stopped at H8 epoch `1/4`, batch `515/10310` (`5.0%`),
  loss `16.426107`, accuracy `0.000061`, throughput `119.8` examples/s. GPU0 used about
  `44369/81559 MiB`, so there was enough memory headroom to test batch `96`.
- Batch-96 retry run id: `H9_official_eres_filtered_pseudo_public_c4_20260414T2113Z`.
- Execution environment: `<redacted>`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=0`.
- Log:
  `artifacts/logs/H9_official_eres_filtered_pseudo_public_c4_20260414T2113Z.log`.
- PID file:
  `artifacts/logs/H9_official_eres_filtered_pseudo_public_c4_20260414T2113Z.pid`.
- Latest pointer:
  `artifacts/logs/latest_H9_official_eres_filtered_pseudo_public_c4.txt`.
- Batch-96 status:
  launched detached on remote GPU0 at `2026-04-14T21:15:24Z` with PID `499100`.
  First H8 training log line reached epoch `1/4`, batch `1/6873`, loss `16.738117`,
  accuracy `0.0`, throughput `30.9` examples/s. GPU0 was active at about
  `65889/81559 MiB` and `100%` utilization, so batch `96` fits with useful headroom.

Completion:

- The batch-96 run completed end-to-end on remote GPU0. H9 checkpoint:
  `artifacts/baselines/official-3dspeaker-eres2net-large-filtered-pseudo-lowlr/20260415T040647Z-1f4c2e26a77f/official_3dspeaker_eres2net_encoder.pt`.
- H9 verification summary on the filtered-pseudo stage:
  EER `0.048481`, minDCF `0.339688`, score gap `0.521726`.
- Public tail output:
  `artifacts/backbone_public/official_3dspeaker_eres2net_large_h9_pseudo_20260414T2113Z/`.
- Public C4 local metrics: `top10_mean_score_mean=0.604332`, `top1_score_mean=0.680532`,
  `label_used_share=0.866834`, Gini@10 `0.299933`, max in-degree `50`.
- Public submission artifacts:
  `artifacts/backbone_public/official_3dspeaker_eres2net_large_h9_pseudo_20260414T2113Z/submission.csv`
  with SHA-256 `720096b64ed24c9246e8e8404d422ead2606861c7ba04730710f1f92ce4c65cd`,
  and
  `artifacts/backbone_public/official_3dspeaker_eres2net_large_h9_pseudo_20260414T2113Z/submission_H9_official_eres_filtered_pseudo_public_c4_20260414T2113Z_c4.csv`.
- User-submitted public leaderboard score on 2026-04-15: `0.5834`.

Decision:

- Reject H9 as a standalone submission branch. It clears MS1 by `+0.0139`, which confirms
  that official/pretrained ERes plus staged pseudo-labeling is not useless, but it remains
  well below `MS31` (`-0.1184`), `MS32` (`-0.1545`), and `MS41` (`-0.1639`).
- Do not spend more direct submission budget on this official ERes staged line in its
  current form. Keep it only as a possible orthogonal backbone for future fusion work if
  later evidence suggests complementary errors.
