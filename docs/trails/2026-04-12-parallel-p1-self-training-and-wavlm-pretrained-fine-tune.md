# 2026-04-12 — Parallel P1 Self-Training and WavLM Pretrained Fine-Tune

Context:

- `G6_p1_clusterfirst_mutual20_shared4_penalty020_top300` scored `0.2369` on public,
  below safe `P1_eres2netv2_h100_b128_public_c4 = 0.2410`.
- Decision: stop graph-only public variants over P1 and use G6 clusters only as a
  pseudo-label pool while testing an orthogonal pretrained representation.

Pseudo-label manifest:

```bash
cd <repo-root>
uv run --group train python scripts/build_pseudo_label_manifests.py \
  --clusters-csv artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g6_penalty/clusters_G6_p1_clusterfirst_mutual20_shared4_penalty020_top300.csv \
  --public-manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --base-train-manifest artifacts/manifests/participants_fixed/train_manifest.jsonl \
  --output-dir artifacts/manifests/pseudo_g6 \
  --experiment-id g6 \
  --min-cluster-size 8 \
  --max-cluster-size 80
```

Manifest result:

- Pseudo rows: `84280`
- Pseudo clusters: `4791`
- Mixed train rows: `744084`
- Pseudo manifest:
  `artifacts/manifests/pseudo_g6/g6_pseudo_manifest.jsonl`
- Mixed manifest:
  `artifacts/manifests/pseudo_g6/g6_mixed_train_manifest.jsonl`

Parallel remote launches:

| Run id | GPU | Hypothesis | Config / command | Status |
| --- | --- | --- | --- | --- |
| `eres2netv2_g6_pseudo_ft_20260412T100724Z` | `CUDA_VISIBLE_DEVICES=0` | P1 self-training: initialize from safe ERes2NetV2 P1 checkpoint, train on original participant train plus filtered G6 pseudo clusters, then recluster/re-run C4 tail after checkpoint is ready. | `configs/training/eres2netv2-g6-pseudo-finetune.toml`; init checkpoint `artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee/eres2netv2_encoder.pt`; `training.batch_size=128`, `max_epochs=3`, mixed crop `2s..6s`, bf16, SGD cosine LR `0.01`, AAM margin `0.2`; log `artifacts/logs/eres2netv2_g6_pseudo_ft_20260412T100724Z.log` | Training completed. Final epoch loss `0.949111`, train acc `0.973660`, LR `1e-5`. Checkpoint `artifacts/baselines/eres2netv2-g6-pseudo-finetune/20260412T100738Z-6b686847f5d8/eres2netv2_encoder.pt`; training summary same dir. The post-train finalizer was manually stopped after checkpoint/dev embeddings because it started generating all-pairs dev verification trials from `13473` dev rows; that local score is not needed for public C4 tail and would waste time/memory. |
| `wavlm_base_plus_sv_ft_20260412T101216Z` | `CUDA_VISIBLE_DEVICES=1` | Orthogonal pretrained encoder: fine-tune `microsoft/wavlm-base-plus-sv` on participant train with raw waveform crops and ArcMargin classifier. This tests a representation family different from CAM++/ERes2NetV2 before fusion. | `configs/training/wavlm-base-plus-sv-participants-finetune.toml`; model provenance Hugging Face `microsoft/wavlm-base-plus-sv`; original participant train manifest only; `batch_size=8`, `steps_per_epoch=2000`, `max_epochs=1`, 4s crops, bf16, AdamW, model LR `2e-5`, classifier LR `1e-3`, feature encoder frozen, gradient checkpointing enabled; log `artifacts/logs/wavlm_base_plus_sv_ft_20260412T101216Z.log` | Completed. `16000` examples, `214.31s`, loss `14.2402`, train acc `0.0000`, embedding size `512`, speaker count `10848`. Saved model dir `artifacts/baselines/wavlm-base-plus-sv-participants-finetune/wavlm_base_plus_sv_ft_20260412T101216Z/hf_model`; checkpoint `hf_xvector_finetune.pt`; metrics `artifacts/tracking/wavlm_base_plus_sv_ft_20260412T101216Z/metrics.jsonl`. Diagnostic note: exact train acc is not useful yet because this was a very short 10.8k-class adaptation run; public tail/fusion decides usefulness. |

Public inference launched from WavLM fine-tuned model:

```bash
cd <repo-root>
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 uv run --group train python scripts/run_hf_xvector_tail.py \
  --model-id artifacts/baselines/wavlm-base-plus-sv-participants-finetune/wavlm_base_plus_sv_ft_20260412T101216Z/hf_model \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_ft_public_c4_20260412T101651Z \
  --experiment-id H2_wavlm_base_plus_sv_ft_public_c4 \
  --batch-size 80 \
  --crop-seconds 6.0 \
  --n-crops 3 \
  --top-cache-k 200 \
  --search-batch-size 2048 \
  --search-device cuda
```

Current H2 status:

- Run id: `wavlm_base_plus_sv_ft_public_c4_20260412T101651Z`
- Log: `artifacts/logs/wavlm_base_plus_sv_ft_public_c4_20260412T101651Z.log`
- Output dir:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_ft_public_c4_20260412T101651Z/`
- Completed. Validator passed; submission:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_ft_public_c4_20260412T101651Z/submission_H2_wavlm_base_plus_sv_ft_public_c4.csv`
- Summary:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_ft_public_c4_20260412T101651Z/H2_wavlm_base_plus_sv_ft_public_c4_summary.json`
- Key diagnostics: `top1_score_mean=0.99985`, `top10_mean_score_mean=0.99985`,
  `indegree_gini_10=0.2457`, max in-degree `35`, label used share `0.8531`.
- Random 2000-row pairwise cosine sample from H2 embeddings: mean `0.9954`, p50
  `0.9968`, p99 `0.9997`.
- Submission overlap with safe P1: mean `0.005/10`, p50 `0`, zero-overlap share
  `0.9951`.

H2 decision:

- Do not submit H2 directly. The validator passed, but the embedding space is nearly
  collapsed after short fine-tuning; the almost-zero overlap with P1 is not useful
  diversity by itself.
- Keep H2 artifact only as a diagnostic showing that naive short WavLM fine-tuning with
  a large 10.8k-class ArcMargin head is unsafe.

Follow-up launch on freed GPU1:

```bash
cd <repo-root>
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 uv run --group train python scripts/run_hf_xvector_tail.py \
  --model-id microsoft/wavlm-base-plus-sv \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_public_c4_20260412T104526Z \
  --experiment-id H1_wavlm_base_plus_sv_pretrained_public_c4 \
  --batch-size 80 \
  --crop-seconds 6.0 \
  --n-crops 3 \
  --top-cache-k 200 \
  --search-batch-size 2048 \
  --search-device cuda
```

- Run id: `wavlm_base_plus_sv_pretrained_public_c4_20260412T104526Z`
- Log: `artifacts/logs/wavlm_base_plus_sv_pretrained_public_c4_20260412T104526Z.log`
- Purpose: separate pretrained WavLM representation quality from the failed short
  fine-tune. If H1 avoids the H2 cosine-collapse pattern, use it for `P1 + WavLM`
  fusion; otherwise move to a different pretrained family.
- Completed. Validator passed; submission:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_public_c4_20260412T104526Z/submission_H1_wavlm_base_plus_sv_pretrained_public_c4.csv`
- Summary:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_public_c4_20260412T104526Z/H1_wavlm_base_plus_sv_pretrained_public_c4_summary.json`
- Key diagnostics: `top1_score_mean=0.9640`, `top10_mean_score_mean=0.9582`,
  `label_count=20360`, `label_used_share=0.7476`, `indegree_gini_10=0.2324`,
  max in-degree `34`.
- Random 2000-row pairwise cosine sample: mean `0.6314`, p50 `0.6453`, p99
  `0.9257`. Unlike H2, H1 is not collapsed.
- Submission overlap with safe P1: mean `0.309/10`, p50 `0`, zero-overlap share
  `0.8183`. This is real orthogonality, but direct public quality is low.
- Public LB score: `0.1228`.

H1 decision:

- Rejected as a direct branch. H1 is only slightly above B8 `0.1223` and below C4
  `0.1249`, despite a very high `top10_mean_score_mean=0.9582`.
- Interpretation: generic pretrained WavLM geometry is not enough by itself; the useful
  WavLM result is the domain-adapted E1 branch (`0.2833`), not raw pretrained inference.
- Keep H1 only as a diagnostic/fusion ingredient, and gate any H1 fusion against P3/E1
  rather than spending further direct public slots on raw pretrained WavLM variants.

NeMo/TitaNet note:

- Attempted import check for TitaNet path: `import nemo.collections.asr` failed with
  `ModuleNotFoundError: No module named 'nemo'`.
- Do not install the NeMo stack mid-run while GPU0 training is active; use a
  Transformers-compatible pretrained speaker model first, then revisit TitaNet setup if
  the current queue is exhausted.

Follow-up launch on GPU1:

```bash
cd <repo-root>
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 uv run --group train python scripts/run_hf_xvector_tail.py \
  --model-id anton-l/wav2vec2-base-superb-sv \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/hf_xvector/wav2vec2_superb_sv_pretrained_public_c4_20260412T111914Z \
  --experiment-id H3_wav2vec2_base_superb_sv_pretrained_public_c4 \
  --batch-size 80 \
  --crop-seconds 6.0 \
  --n-crops 3 \
  --top-cache-k 200 \
  --search-batch-size 2048 \
  --search-device cuda
```

- Run id: `wav2vec2_superb_sv_pretrained_public_c4_20260412T111914Z`
- Log: `artifacts/logs/wav2vec2_superb_sv_pretrained_public_c4_20260412T111914Z.log`
- Purpose: second orthogonal pretrained encoder using the same reproducible HF xvector
  tail, without NeMo dependency risk.
- Completed. Validator passed; submission:
  `artifacts/backbone_public/hf_xvector/wav2vec2_superb_sv_pretrained_public_c4_20260412T111914Z/submission_H3_wav2vec2_base_superb_sv_pretrained_public_c4.csv`
- Summary:
  `artifacts/backbone_public/hf_xvector/wav2vec2_superb_sv_pretrained_public_c4_20260412T111914Z/H3_wav2vec2_base_superb_sv_pretrained_public_c4_summary.json`
- Key diagnostics: `top1_score_mean=0.9725`, `top10_mean_score_mean=0.9695`,
  `label_count=24383`, `label_used_share=0.6948`, `indegree_gini_10=0.2418`,
  max in-degree `34`.
- Random 2000-row pairwise cosine sample: mean `0.7974`, p50 `0.8111`, p99
  `0.9508`. H3 is not as collapsed as H2, but it is more compressed than H1.
- Submission overlap with safe P1: mean `0.163/10`, p50 `0`, zero-overlap share
  `0.8811`.
- Decision: do not spend a public LB slot on H3 direct. Keep as diagnostic/fusion
  material only.

Local artifact sync note:

- The standard repo sync excludes `artifacts/`, so lightweight H1/H2/H3 summary,
  validation, and submission files were copied back manually. Large `.npy` embedding
  arrays remain remote-only unless needed locally.

Heavier GPU1 utilization launch:

```bash
cd <repo-root>
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 uv run --group train python scripts/run_hf_xvector_tail.py \
  --model-id microsoft/wavlm-base-plus-sv \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_8s5c_b128_public_c4_20260412T114046Z \
  --experiment-id H4_wavlm_base_plus_sv_pretrained_8s5c_b128_public_c4 \
  --batch-size 128 \
  --crop-seconds 8.0 \
  --n-crops 5 \
  --top-cache-k 300 \
  --search-batch-size 4096 \
  --search-device cuda
```

- Run id: `wavlm_base_plus_sv_pretrained_8s5c_b128_public_c4_20260412T114046Z`
- Log: `artifacts/logs/wavlm_base_plus_sv_pretrained_8s5c_b128_public_c4_20260412T114046Z.log`
- Purpose: use more of GPU1 safely and test whether longer/more crops improve the
  strongest orthogonal pretrained candidate H1 before public LB spend.
- Monitor at 2026-04-12 11:57 UTC: extraction at `40%`; GPU1 used ~`22.3 GiB`, but
  util only ~`31%`, suggesting decode/IO/CPU-loop bottleneck despite larger batch.
- Completed. Validator passed; submission:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_8s5c_b128_public_c4_20260412T114046Z/submission_H4_wavlm_base_plus_sv_pretrained_8s5c_b128_public_c4.csv`
- Summary:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_8s5c_b128_public_c4_20260412T114046Z/H4_wavlm_base_plus_sv_pretrained_8s5c_b128_public_c4_summary.json`
- Diagnostics: `top1_score_mean=0.9635`, `top10_mean_score_mean=0.9576`,
  `label_used_share=0.7413`, random pairwise cosine mean `0.6347`, p50 `0.6491`,
  p99 `0.9245`.
- Overlap with safe P1: mean `0.313/10`, p50 `0`, zero-overlap share `0.8186`.
- Overlap with H1: mean `2.785/10`, p50 `3`.
- Decision: H4 does not materially change the H1 picture. Keep as a slightly heavier
  WavLM fusion candidate; do not submit direct.

Additional concurrent GPU1 launch:

```bash
cd <repo-root>
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 uv run --group train python scripts/run_hf_xvector_tail.py \
  --model-id anton-l/wav2vec2-base-superb-sv \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/hf_xvector/wav2vec2_superb_sv_pretrained_8s5c_b128_public_c4_20260412T115810Z \
  --experiment-id H5_wav2vec2_base_superb_sv_pretrained_8s5c_b128_public_c4 \
  --batch-size 128 \
  --crop-seconds 8.0 \
  --n-crops 5 \
  --top-cache-k 300 \
  --search-batch-size 4096 \
  --search-device cuda
```

- Run id: `wav2vec2_superb_sv_pretrained_8s5c_b128_public_c4_20260412T115810Z`
- Log: `artifacts/logs/wav2vec2_superb_sv_pretrained_8s5c_b128_public_c4_20260412T115810Z.log`
- Purpose: keep GPU1 closer to full utilization while still below the requested
  no-OOM VRAM target. At launch with H4 concurrent: GPU1 ~`44.7 GiB`, 100% util.
- Completed. Validator passed; submission:
  `artifacts/backbone_public/hf_xvector/wav2vec2_superb_sv_pretrained_8s5c_b128_public_c4_20260412T115810Z/submission_H5_wav2vec2_base_superb_sv_pretrained_8s5c_b128_public_c4.csv`
- Summary:
  `artifacts/backbone_public/hf_xvector/wav2vec2_superb_sv_pretrained_8s5c_b128_public_c4_20260412T115810Z/H5_wav2vec2_base_superb_sv_pretrained_8s5c_b128_public_c4_summary.json`
- Diagnostics: `top1_score_mean=0.9724`, `top10_mean_score_mean=0.9695`,
  `label_used_share=0.6805`, `indegree_gini_10=0.2457`, max in-degree `34`,
  same-label candidates p50 `5`, p95 `19`.
- Public LB decision: do not submit H4/H5 direct. H5 mostly confirms the earlier H3
  compressed Wav2Vec2 geometry and is not a safer direct candidate than P1.

E1 domain WavLM fine-tune launch:

- Hypothesis: the useful next jump should come from pretrained-first WavLM adaptation,
  not from direct pretrained inference or another graph-only P1 variant. The previous
  short WavLM fine-tune (`H2`) collapsed because LR/run length were too aggressive and
  too underdeveloped for a 10.8k-class ArcMargin head.
- Code/config changes:
  - `src/kryptonite/training/hf_xvector.py` now supports train-only mixed crop lengths
    plus domain augmentations before HF feature extraction: random bandlimit, leading /
    trailing silence, mild peak limiting, random gain, and light Gaussian noise.
  - Config:
    `configs/training/wavlm-base-plus-sv-e1-domain-finetune.toml`
  - Checks before sync: `ruff check`, `ty check`, and `pytest tests/unit/test_eda.py -q`
    passed for the touched WavLM training path.
- Remote sync: repo synced to `remote` with the standard rsync command excluding
  `.venv/`, `.cache/`, `datasets/`, and general `artifacts/`.
- Run id: `wavlm_e1_domain_ft_20260412T130254Z`
- GPU: `CUDA_VISIBLE_DEVICES=1`
- Log: `artifacts/logs/wavlm_e1_domain_ft_20260412T130254Z.log`
- PID file: `artifacts/logs/wavlm_e1_domain_ft_20260412T130254Z.pid`
- Output root:
  `artifacts/baselines/wavlm-base-plus-sv-e1-domain-finetune/wavlm_e1_domain_ft_20260412T130254Z/`
- Metrics path:
  `artifacts/tracking/wavlm_e1_domain_ft_20260412T130254Z/metrics.jsonl`
- Exact launch command:

```bash
cd <repo-root>
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
uv run --group train python scripts/run_hf_xvector_finetune.py \
  --config configs/training/wavlm-base-plus-sv-e1-domain-finetune.toml \
  --run-id wavlm_e1_domain_ft_20260412T130254Z \
  --device cuda
```

- Training recipe: Hugging Face `microsoft/wavlm-base-plus-sv`, original participant
  train manifest `artifacts/manifests/participants_fixed/train_manifest.jsonl`, seed
  `42`, `batch_size=96`, `steps_per_epoch=8000`, `max_epochs=3`, mixed crops `2s..6s`,
  bf16, AdamW, model LR `3e-6`, classifier LR `5e-4`, cosine schedule, warmup `1000`,
  feature encoder frozen, gradient checkpointing enabled, ArcMargin scale `32`, margin
  `0.2`.
- Augmentation recipe: bandlimit p `0.45`, edge silence p `0.45` with leading up to
  `0.8s` and trailing up to `1.4s`, peak limiter p `0.20`, gain p `0.25` in
  `[-3, +2] dB`, Gaussian noise p `0.15` with SNR `18..35 dB`.
- First monitor: epoch `1/3`, step `200/8000`, loss `16.7273`, train acc `0.000156`,
  LR `6e-7`, throughput `162.8` examples/s; GPU1 ~`18.7 GiB`, 93-98% util. VRAM is
  below the allowed 80% cap, but compute utilization is already high, so the first E1
  pass is kept running rather than restarted for a larger batch.
- Epoch 1 completed: `768000` examples, `8000` steps, train loss `10.5764`, train acc
  `0.2428`, epoch seconds `4770.59`, LR `2.3968e-6`. Metrics persisted in
  `artifacts/tracking/wavlm_e1_domain_ft_20260412T130254Z/metrics.jsonl` and
  `training_summary.json`.
- Follow-up monitor shortly after epoch 2 start: step `200/8000`, loss `7.2987`, train
  acc `0.5120`, throughput ~`165` examples/s, GPU1 ~`19.2 GiB`, util ~`91%`.
- Epoch 2 completed: `768000` examples, `8000` steps, train loss `6.4776`, train acc
  `0.5751`, epoch seconds `4484.56`, LR `9.1941e-7`. Metrics persisted in
  `artifacts/tracking/wavlm_e1_domain_ft_20260412T130254Z/metrics.jsonl` and
  `training_summary.json`.
- Follow-up monitor shortly after epoch 3 start: step `500/8000`, loss `5.6425`, train
  acc `0.6495`, throughput ~`169` examples/s, GPU1 ~`19.4 GiB`, util `100%`.
- Epoch 3 completed: `768000` examples, `8000` steps, train loss `5.4627`, train acc
  `0.6662`, epoch seconds `4479.97`, LR `1.5e-7`. Metrics persisted in
  `artifacts/tracking/wavlm_e1_domain_ft_20260412T130254Z/metrics.jsonl` and
  `training_summary.json`.
- Saved model:
  `artifacts/baselines/wavlm-base-plus-sv-e1-domain-finetune/wavlm_e1_domain_ft_20260412T130254Z/hf_model/`
- Saved checkpoint:
  `artifacts/baselines/wavlm-base-plus-sv-e1-domain-finetune/wavlm_e1_domain_ft_20260412T130254Z/hf_xvector_finetune.pt`
- Decision: training completed cleanly and GPU1 was freed. Per operator instruction,
  launch exactly one public C4 tail on GPU1, then stop further GPU work until the public
  LB result is known.

E1 domain WavLM public C4 tail:

- Hypothesis: after domain fine-tuning, the WavLM speaker geometry may become useful
  enough for a direct public C4 submission and later P3+WavLM fusion if public/local
  diagnostics are promising.
- Run id: `wavlm_e1_domain_ft_public_c4_20260412T165411Z`
- GPU: `CUDA_VISIBLE_DEVICES=1`
- Log:
  `artifacts/logs/wavlm_e1_domain_ft_public_c4_20260412T165411Z.log`
- PID file:
  `artifacts/logs/wavlm_e1_domain_ft_public_c4_20260412T165411Z.pid`
- Model dir:
  `artifacts/baselines/wavlm-base-plus-sv-e1-domain-finetune/wavlm_e1_domain_ft_20260412T130254Z/hf_model/`
- Output dir:
  `artifacts/backbone_public/hf_xvector/wavlm_e1_domain_ft_public_c4_20260412T130254Z/`
- Exact launch command:

```bash
cd <repo-root>
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
uv run --group train python scripts/run_hf_xvector_tail.py \
  --model-id artifacts/baselines/wavlm-base-plus-sv-e1-domain-finetune/wavlm_e1_domain_ft_20260412T130254Z/hf_model \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv "datasets/Для участников/test_public.csv" \
  --output-dir artifacts/backbone_public/hf_xvector/wavlm_e1_domain_ft_public_c4_20260412T130254Z \
  --experiment-id E1_wavlm_domain_ft_public_c4 \
  --batch-size 128 \
  --crop-seconds 6.0 \
  --n-crops 3 \
  --top-cache-k 300 \
  --search-batch-size 4096 \
  --search-device cuda
```

- Initial monitor: model loaded, extraction started on `134697` public rows; GPU1
  ~`15.9 GiB`, util `97%`. GPU0 remains occupied by an external process and is not used.
- Completed. Extraction took `1232.73s`; exact top-k search took `0.74s`; C4-style label
  propagation/rerank took `9.77s`. Validator passed with `134697/134697` rows, `K=10`,
  and `0` errors.
- Submission:
  `artifacts/backbone_public/hf_xvector/wavlm_e1_domain_ft_public_c4_20260412T130254Z/submission_E1_wavlm_domain_ft_public_c4.csv`
- Summary:
  `artifacts/backbone_public/hf_xvector/wavlm_e1_domain_ft_public_c4_20260412T130254Z/E1_wavlm_domain_ft_public_c4_summary.json`
- Validation:
  `artifacts/backbone_public/hf_xvector/wavlm_e1_domain_ft_public_c4_20260412T130254Z/submission_E1_wavlm_domain_ft_public_c4_validation.json`
- Diagnostics: `top1_score_mean=0.8552`, `top10_mean_score_mean=0.8124`,
  `label_used_share=0.7778`, Gini@10 `0.2649`, max in-degree `47`, same-label
  candidates p50 `6`, p95 `29`, reciprocal share `0.1660`.
- Lightweight CSV/JSON artifacts were synced back locally for public LB upload.
- Public LB score: `0.2833`.
- Decision: direct E1 is slightly below the current safe P3 score `0.2861` by `0.0028`,
  so it does not replace P3. The score is close enough, and the backbone is orthogonal
  enough, to keep E1 for a later P3+WavLM fusion branch. Per operator instruction,
  stop here: do not start fusion, another tail, or another training job now. GPU1 was
  free after the run; GPU0 was not used.

P3 public tail from P1 pseudo-fine-tuned checkpoint:

- Hypothesis: if G6 pseudo clusters contain enough clean same-speaker structure, the
  P1 initialized encoder fine-tuned on mixed real+pseudo labels should improve public C4
  retrieval or at least become a useful fusion candidate.
- Source checkpoint:
  `artifacts/baselines/eres2netv2-g6-pseudo-finetune/20260412T100738Z-6b686847f5d8/eres2netv2_encoder.pt`
- Run id: `p3_eres2netv2_g6_pseudo_ft_public_c4_20260412T132640Z`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- Log:
  `artifacts/logs/p3_eres2netv2_g6_pseudo_ft_public_c4_20260412T132640Z.log`
- PID file:
  `artifacts/logs/p3_eres2netv2_g6_pseudo_ft_public_c4_20260412T132640Z.pid`
- Output dir:
  `artifacts/backbone_public/eres2netv2_g6_pseudo_ft/20260412T100738Z-6b686847f5d8/`
- Exact launch command:

```bash
cd <repo-root>
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
uv run --group train python scripts/run_torch_checkpoint_c4_tail.py \
  --model eres2netv2 \
  --checkpoint-path artifacts/baselines/eres2netv2-g6-pseudo-finetune/20260412T100738Z-6b686847f5d8/eres2netv2_encoder.pt \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/eres2netv2_g6_pseudo_ft/20260412T100738Z-6b686847f5d8 \
  --experiment-id P3_eres2netv2_g6_pseudo_ft_public_c4 \
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

- First monitor: started embedding extraction on `134697` public rows; GPU0 ~`21.9 GiB`,
  util ~`64%`; validator/submission pending.
- Completed. Validator passed; submission:
  `artifacts/backbone_public/eres2netv2_g6_pseudo_ft/20260412T100738Z-6b686847f5d8/submission_P3_eres2netv2_g6_pseudo_ft_public_c4.csv`
- Summary:
  `artifacts/backbone_public/eres2netv2_g6_pseudo_ft/20260412T100738Z-6b686847f5d8/P3_eres2netv2_g6_pseudo_ft_public_c4_summary.json`
- Runtime: embedding extraction `2819.30s`, top-k search `0.77s`, C4 rerank `11.27s`.
- Retrieval diagnostics: `top1_score_mean=0.6901`, `top10_mean_score_mean=0.6116`,
  label count `12103`, label used share `0.8297`, reciprocal share `0.1836`,
  same-label candidates p50 `8`, p95 `26`.
- Hubness/overlap vs safe P1:
  - P3 `Gini@10=0.2810`, max in-degree `44`;
  - safe P1 `Gini@10=0.3510`, max in-degree `69`;
  - mean overlap vs P1 `4.99/10`, p50 `5`, p10 `2`, p90 `8`;
  - top1 same vs P1 `0.4890`, zero-overlap share `0.0134`.
- Public LB: `0.2861`.
- Decision: new public best and new safe branch. The lower cosine score scale was not a
  blocker; lower hubness plus the pseudo-label adapted encoder transferred to public.
  Pseudo-label self-training over filtered G6 clusters is now confirmed as a productive
  direction.
- Lightweight P3 summary, validation, and submission CSV were synced back locally under
  the same `artifacts/backbone_public/eres2netv2_g6_pseudo_ft/...` path; large embedding
  `.npy` remains remote-only.

Aborted follow-up:

- After the P3 public gain was known, a cheap P3+P1 rank-score fusion sweep
  (`F2/F3/F4`) was briefly launched on GPU0 with
  `scripts/run_backbone_fusion_c4_tail.py`.
- The user then requested GPU0 be left free for their own work. The fusion sweep was
  stopped before producing a completed summary/submission, and GPU0 was confirmed free
  (`0 MiB`, `0% util`). No decision should be drawn from the partial fusion output.
