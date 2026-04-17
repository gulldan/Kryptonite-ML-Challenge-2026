# 2026-04-13 — Official Pretrained WavLM and ERes2Net-Large Probes

H6 WavLM official-HF no-trim public C4:

- Hypothesis: the earlier raw WavLM public score `0.1228` may have been hurt by this
  repository's silence trim policy rather than the pretrained model itself.
- Model: Hugging Face `microsoft/wavlm-base-plus-sv`; official Transformers class
  `AutoModelForAudioXVector`; feature frontend `AutoFeatureExtractor`.
- Command:

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 uv run --group train python scripts/run_hf_xvector_tail.py \
  --model-id microsoft/wavlm-base-plus-sv \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_official_notrim_6s3c_public_c4_20260413T \
  --experiment-id H6_wavlm_base_plus_sv_pretrained_official_notrim_6s3c_public_c4 \
  --batch-size 80 \
  --crop-seconds 6.0 \
  --n-crops 3 \
  --no-trim \
  --top-cache-k 200 \
  --search-batch-size 2048 \
  --search-device cuda
```

- Completed locally on RTX 4090. Embedding runtime `1257.17s`; search `0.999s`; rerank
  `6.579s`.
- Validator passed, `0` errors, `134697` rows.
- Submission:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_official_notrim_6s3c_public_c4_20260413T/submission_H6_wavlm_base_plus_sv_pretrained_official_notrim_6s3c_public_c4.csv`.
- Short copy for upload:
  `artifacts/backbone_public/hf_xvector/submission_H6_wavlm_base_plus_sv_pretrained_official_notrim_6s3c_public_c4.csv`.
- Summary:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_official_notrim_6s3c_public_c4_20260413T/H6_wavlm_base_plus_sv_pretrained_official_notrim_6s3c_public_c4_summary.json`.
- Diagnostics: `top1_score_mean=0.9646`, `top10_mean_score_mean=0.9590`,
  `label_used_share=0.7489`, Gini@10 `0.2321`, max in-degree `34`.
- Comparison against H1 trim run: mean overlap@10 `2.970`, median `3`, top1 equal
  `21.93%`. H6 is a materially different no-trim ranking, but public expectation remains
  low until LB is checked because H1 scored only `0.1228`.

H7 official 3D-Speaker ERes2Net-large public C4 launch:

- Hypothesis: a clean pretrained 3D-Speaker ERes2Net-large may transfer better than the
  from-scratch ERes2NetV2/CAM++ participant checkpoints and should be tested analogously
  to the successful ModelScope CAM++ default branch.
- Official source: 3D-Speaker repository, ERes2Net-large model id
  `iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k`; 3D-Speaker README lists
  ERes2Net-large as `22.46M` parameters and the official model id.
- remote preparation: cloned 3D-Speaker to `/tmp/3D-Speaker`; ModelScope downloaded
  `eres2net_large_model.ckpt` under
  `/root/.cache/modelscope/hub/models/iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k/`.
- Decoder note: official `infer_sv_batch.py` uses `torchaudio.load`, but the remote
  environment's torchaudio requires TorchCodec/FFmpeg libraries that are unavailable for
  FLAC decode. Smoke showed `soundfile` can read the same FLAC paths. Added
  `scripts/run_official_3dspeaker_eres2net_tail.py`, which keeps official 3D-Speaker
  ERes2Net architecture, weights, FBank, 10s circular chunking, and mean segment pooling,
  replacing only the broken audio decoder with `soundfile`.
- Smoke: 3-file extraction passed on remote GPU1 with the new runner; C4 smoke failed only
  because top-cache was intentionally too small for the full label-propagation config.
- Initial remote run id: `H7_eres2net_large_3dspeaker_pretrained_public_c4_20260413T0400Z` (stopped before 5% because batch `32` was too slow).
- Active remote run id: `H7b_eres2net_large_3dspeaker_pretrained_public_c4_b80_20260413T0408Z`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=1` on `remote` inside container `container`.
- Log:
  `artifacts/logs/H7b_eres2net_large_3dspeaker_pretrained_public_c4_b80_20260413T0408Z.log`.
- Output directory:
  `artifacts/backbone_public/official_3dspeaker_eres2net_large_20260413T/full_b80/`.
- Command:

```bash
cd <repo-root>
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 uv run --with scipy \
  python scripts/run_official_3dspeaker_eres2net_tail.py \
  --checkpoint-path artifacts/modelscope_cache/official_3dspeaker/speech_eres2net_large_sv_zh-cn_3dspeaker_16k/eres2net_large_model.ckpt \
  --speakerlab-root /tmp/3D-Speaker \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv artifacts/links/participants_dataset/test_public.csv \
  --data-root artifacts/links/participants_dataset \
  --output-dir artifacts/backbone_public/official_3dspeaker_eres2net_large_20260413T/full_b80 \
  --experiment-id H7b_eres2net_large_3dspeaker_pretrained_public_c4_b80_20260413T0408Z \
  --device cuda \
  --search-device cuda \
  --batch-size 80 \
  --search-batch-size 2048 \
  --top-cache-k 200
```

- Status at launch: running; active batch-80 log line processed `1/134697` rows, GPU1 memory around `48.6 GiB` with `100%` utilization. Public score pending.
