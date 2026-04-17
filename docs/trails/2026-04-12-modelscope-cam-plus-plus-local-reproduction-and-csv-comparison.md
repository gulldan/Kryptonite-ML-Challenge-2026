# 2026-04-12 — ModelScope CAM++ Local Reproduction And CSV Comparison

Context:

- User provided the scored default ModelScope CAM++ submission:
  `artifacts/backbone_public/campp/default_model_submission.csv`, public LB `0.5695`.
- Goal: run the same ModelScope CAM++ VoxCeleb checkpoint locally through the current repo
  inference path and compare the produced ranking against the scored CSV.

Checkpoint preparation:

- Source ModelScope checkpoint:
  `artifacts/modelscope_cache/iic/speech_campplus_sv_en_voxceleb_16k/campplus_voxceleb.bin`.
- Local converted checkpoint:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/converted/modelscope_campplus_voxceleb_encoder.pt`.
- Conversion was a key remap only; tensor values were unchanged. The ModelScope
  `state_dict` and local `CAMPPlusEncoder` both contain `937` keys with matching tensor
  shapes after remap.
- Conversion report:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/converted/conversion_summary.json`.

Local run:

```bash
uv run python scripts/run_torch_checkpoint_c4_tail.py \
  --model campp \
  --checkpoint-path artifacts/backbone_public/modelscope_campplus_voxceleb_default/converted/modelscope_campplus_voxceleb_encoder.pt \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/modelscope_campplus_voxceleb_default/20260412T_local_ms2_c4 \
  --experiment-id MS2_modelscope_campplus_voxceleb_default_public_c4 \
  --device cuda \
  --search-device cuda \
  --precision bf16 \
  --batch-size 512 \
  --search-batch-size 2048 \
  --top-cache-k 100 \
  --crop-seconds 6.0 \
  --n-crops 3 \
  --trim \
  --shift-mode none
```

Outputs:

- C4 submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/20260412T_local_ms2_c4/submission_MS2_modelscope_campplus_voxceleb_default_public_c4.csv`
- Exact-only submission from the same local embeddings:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/20260412T_local_ms2_c4/submission_MS2_modelscope_campplus_voxceleb_default_public_exact.csv`
- Comparison report:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/20260412T_local_ms2_c4/default_vs_local_comparison.json`

Validation and runtime:

- MS1 scored CSV validation: passed, `0` errors.
- MS2 local exact validation: passed.
- MS2 local C4 validation: passed.
- Local extraction runtime on RTX 4090: `625.99` seconds for `134697` rows.
- Search runtime: `0.97` seconds. C4 rerank runtime: `7.01` seconds.

Comparison against the scored MS1 CSV:

| Compared files | First-row neighbors | Top-1 match | Mean overlap@10 | Median overlap@10 | Same ordered row share |
| --- | --- | ---: | ---: | ---: | ---: |
| MS1 scored vs MS2 local exact | MS1 `1437,24932,75809,37108,39021,124530,117542,76244,8574,90474`; local exact `37815,29484,39021,99711,21757,134289,12635,74352,76244,59815` | `15.86%` | `2.459/10` | `2` | `0.0%` |
| MS1 scored vs MS2 local C4 | MS1 same as above; local C4 `39021,134289,74352,59815,46821,37815,99711,29484,21757,76244` | `15.41%` | `2.510/10` | `2` | `0.0%` |
| MS2 local exact vs MS2 local C4 | local exact vs local C4 | `69.10%` | `6.610/10` | `7` | `0.21%` |

Additional frontend ablation:

- Run id: `MS3_modelscope_campplus_voxceleb_default_public_notrim_1crop_c4`.
- Command delta vs MS2: `--no-trim --n-crops 1 --crop-seconds 6.0`.
- Output directory:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/20260412T_local_ms3_notrim_1crop`.
- Comparison report:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/20260412T_local_ms3_notrim_1crop/default_vs_ms3_notrim_1crop_comparison.json`.

| Compared files | First-row neighbors | Top-1 match | Mean overlap@10 | Median overlap@10 | Same ordered row share |
| --- | --- | ---: | ---: | ---: | ---: |
| MS1 scored vs MS3 no-trim exact | MS1 same as above; MS3 exact `21757,59815,37815,76599,126291,125785,27220,803,18136,1436` | `11.92%` | `2.075/10` | `1` | `0.0%` |
| MS1 scored vs MS3 no-trim C4 | MS1 same as above; MS3 C4 `21757,30197,18136,36063,13519,113047,76599,37815,27220,99711` | `11.90%` | `2.190/10` | `1` | `0.0%` |

Hubness:

- MS1 scored CSV: Gini@10 `0.4917`, max in-degree `214`.
- MS2 local exact: Gini@10 `0.4928`, max in-degree `159`.
- MS2 local C4: Gini@10 `0.3397`, max in-degree `64`.

Embedding diagnostics:

- Local MS2 embeddings path:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/20260412T_local_ms2_c4/embeddings_MS2_modelscope_campplus_voxceleb_default_public_c4.npy`.
- Local MS3 embeddings path:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/20260412T_local_ms3_notrim_1crop/embeddings_MS3_modelscope_campplus_voxceleb_default_public_notrim_1crop_c4.npy`.
- Both local embedding arrays are `float32`, shape `[134697, 512]`, contain `0` NaNs,
  and are L2-normalized with norm p50 `1.0`.
- Row-wise cosine between MS2 and MS3 embeddings: mean `0.9469`, p50 `0.9690`, p05
  `0.8440`, p95 `1.0`. This means the local trim/crop ablation changes embeddings
  moderately but does not explain the much larger ranking mismatch against MS1.
- Embedding diagnostic report:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/local_embedding_diagnostics.json`.

Conclusion:

- The scored MS1 CSV and the local MS2 run are format-compatible and share the same
  checkpoint weights, but their neighbor rankings differ materially.
- This rules out a generic submission writer problem for this comparison. MS3 also shows
  that simply disabling trim and switching to one center crop does not recover the scored
  ranking.
- The gap is in the inference/frontend policy: official/default ModelScope preprocessing,
  fbank details, segment/full-utterance behavior, or normalization differs from the repo's
  current local fbank path.
- The scored MS1 CSV should remain the safe public artifact until the exact ModelScope
  inference path is reproduced locally. MS2 is useful only as a diagnostic and should not
  be submitted as a replacement without further frontend parity work.
