# 2026-04-12 — Official GitHub CAM++ Reproduction

Context:

- User shared the exact code repo used for the `0.5695` submission:
  `https://github.com/RustamOper05/kryptonite_tembr_research`.
- Access was verified through `gh`; repo is private and default branch is `master`.
- Clone path for inspection:
  `/tmp/kryptonite_tembr_research`.

Relevant code path:

- Submission builder:
  `/tmp/kryptonite_tembr_research/code/campp/build_submission.py`.
- Retrieval/frontend:
  `/tmp/kryptonite_tembr_research/code/campp/retrieval.py`.
- Model construction:
  `/tmp/kryptonite_tembr_research/code/campp/common.py`.
- Config:
  `/tmp/kryptonite_tembr_research/code/campp/configs/campp_en_ft.base.yaml`.

Important differences from this repository's local MS2/MS3 path:

- The GitHub repo uses official 3D-Speaker code at commit
  `065629c313eaf1a01c65c640c46d77e61e9607b4`:
  `speakerlab.models.campplus.DTDNN.CAMPPlus`.
- It extracts features with `torchaudio.compliance.kaldi.fbank(..., dither=0.0)` and
  applies utterance cepstral mean normalization:
  `features - features.mean(dim=0, keepdim=True)`.
- Its `segment_mean` policy uses 6s segments, repeats short clips, and for files longer
  than 6s averages 3 evenly spaced segment embeddings. It does not use this repository's
  conservative silence trim.
- This repository's failed reproduction used local `FbankExtractor` plus local
  `CAMPPlusEncoder`, so the same checkpoint weights saw materially different features.

Reproduction command:

```bash
PYTHONPATH=/tmp/kryptonite_tembr_research/code/campp \
uv run --with requests --with pandas --with pyarrow --with PyYAML --with soundfile --with scipy --with tqdm \
  python /tmp/kryptonite_tembr_research/code/campp/build_submission.py \
  --config /tmp/kryptonite_tembr_research/code/campp/configs/campp_en_ft.base.yaml \
  --mode segment_mean \
  --csv '/tmp/kryptonite_tembr_research/data/Для участников/test_public.csv' \
  --topk 10 \
  --run-name reproduction_pretrained_segment_mean_test_public \
  --save-embeddings
```

Copied artifacts:

- Reproduced submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/official_repo_reproduction_20260412T2250/submission.csv`.
- Official-repo embeddings:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/official_repo_reproduction_20260412T2250/embeddings.npy`.
- Comparison report:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/official_repo_reproduction_20260412T2250/default_vs_official_repo_reproduction.json`.

Results:

- Reproduction validator: passed, `0` errors.
- Runtime: embedding `666.13s`, search `0.89s`, total `667.02s` on local RTX 4090.
- Reproduced first row exactly matches MS1:
  `1437,24932,75809,37108,39021,124530,117542,76244,8574,90474`.
- MS1 scored CSV vs official-repo reproduction:
  - top1 match `99.53%`;
  - ordered cell equality `96.19%`;
  - exact same ordered row share `81.22%`;
  - same neighbor set share `96.13%`;
  - mean overlap@10 `9.961/10`;
  - median overlap@10 `10`;
  - rows with full overlap@10: `129489/134697`.
- The small non-identical tail is consistent with dependency/runtime top-k tie or minor
  numeric differences; it is not a semantic mismatch.

Embedding comparison:

- Official-repo embeddings are `float32`, shape `[134697, 512]`, contain `0` NaNs, and
  are not stored L2-normalized: norm p50 `19.092`, min `10.697`, max `41.330`.
- Retrieval normalizes them inside `topk_indices_from_embeddings()`.
- Row-wise cosine between official-repo embeddings and local MS2 embeddings:
  mean `0.6114`, p50 `0.6315`, p05 `0.3067`, p95 `0.8499`.
- Row-wise cosine between official-repo embeddings and local MS3 embeddings:
  mean `0.6129`, p50 `0.6318`, p05 `0.3197`, p95 `0.8420`.

Conclusion:

- The `0.5695` result is reproducible from the GitHub code path.
- The issue is not submission formation. The issue is that this repository's local CAM++
  inference path produces different embeddings from the official 3D-Speaker/ModelScope
  frontend.
- For the ModelScope CAM++ branch, future work should import or faithfully reproduce the
  official frontend/model path before applying C4, fusion, pseudo-labeling, or fine-tuning.
