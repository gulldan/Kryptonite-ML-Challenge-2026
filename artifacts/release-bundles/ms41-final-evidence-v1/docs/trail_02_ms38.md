# 2026-04-15 — MS38 Official CAM++ Weight-Space Soup Launch

Hypothesis:

- MS30, MS31, and MS32 are close official-CAM++ descendants of the same ModelScope
  initialization, but they encode slightly different supervised/pseudo-label decisions.
- Late MS32 pseudo refinements improved local public-graph diagnostics while hurting hidden
  public LB, so another pseudo-stage is risky. A convex weight-space average can smooth the
  encoder geometry while still delivering one checkpoint and the existing MS32 C4 tail.
- The first cheap pass should test MS31/MS32 interpolation around the stronger MS32 branch,
  plus MS30/MS31/MS32 weighted soups anchored on MS32. A CN-Celeb/SWA pass is deferred until
  MS37 produces a ready CAM++ checkpoint.

Code change:

- Added `scripts/build_campp_weight_soup.py`, which loads compatible CAM++ encoder
  checkpoints, verifies matching `model_config`, state keys, and tensor shapes, averages
  floating tensors with normalized convex weights, copies non-floating buffers from a
  reference source, and writes a normal `campp_encoder.pt` plus soup metadata.
- Added `tests/unit/test_campp_weight_soup.py`.

Local verification:

```bash
uv run ruff format scripts/build_campp_weight_soup.py tests/unit/test_campp_weight_soup.py
uv run ruff check scripts/build_campp_weight_soup.py tests/unit/test_campp_weight_soup.py
uv run pytest tests/unit/test_campp_weight_soup.py
```

Result: `2` tests passed and targeted ruff check passed.

Source checkpoints:

- MS30:
  `artifacts/baselines/campp-ms1-official-participants-lowlr/20260413T202123Z-d45cc7d9936e/campp_encoder.pt`.
- MS31:
  `artifacts/baselines/campp-ms1-official-participants-voxblink2-augment/20260413T203245Z-b87036ccb3db/campp_encoder.pt`.
- MS32:
  `artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt`.

Planned remote candidates:

| Candidate | Weights | Purpose |
| --- | --- | --- |
| `MS38a_i065_campp_ms31_ms32_interp_20260415T0531Z` | MS31 `0.35`, MS32 `0.65` | lower-MS32 interpolation control |
| `MS38a_i075_campp_ms31_ms32_interp_20260415T0531Z` | MS31 `0.25`, MS32 `0.75` | mid interpolation |
| `MS38a_i085_campp_ms31_ms32_interp_20260415T0531Z` | MS31 `0.15`, MS32 `0.85` | MS32-dominant interpolation |
| `MS38a_i095_campp_ms31_ms32_interp_20260415T0531Z` | MS31 `0.05`, MS32 `0.95` | near-MS32 smoothing |
| `MS38b_soup_ms30_ms31_ms32_w10_25_65_20260415T0531Z` | MS30 `0.10`, MS31 `0.25`, MS32 `0.65` | greedy-soup style three-way candidate |
| `MS38b_soup_ms30_ms31_ms32_w05_20_75_20260415T0531Z` | MS30 `0.05`, MS31 `0.20`, MS32 `0.75` | more conservative three-way candidate |

Remote execution plan:

- Execution environment: `<redacted>`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=1`.
- Output root:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms38_weight_soup_20260415T0531Z`.
- Soup checkpoint root:
  `artifacts/baselines/campp-ms38-weight-soup/20260415T0531Z`.
- Log:
  `artifacts/logs/MS38_campp_weight_soup_public_c4_20260415T0531Z.log`.
- PID file:
  `artifacts/logs/MS38_campp_weight_soup_public_c4_20260415T0531Z.pid`.
- Latest pointer:
  `artifacts/logs/latest_MS38_campp_weight_soup_public_c4.txt`.
- Tail settings: `scripts/run_official_campp_tail.py`, torch backend, packed official
  public frontend cache, `mode=segment_mean`, `eval_chunk_seconds=6.0`,
  `segment_count=3`, `top_cache_k=200`, C4 defaults matching MS32.

Remote launch command:

```bash
cd <repo-root>
RUN_STAMP=20260415T0531Z
BASE_OUT=artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms38_weight_soup_${RUN_STAMP}
SOUP_ROOT=artifacts/baselines/campp-ms38-weight-soup/${RUN_STAMP}
LOG=artifacts/logs/MS38_campp_weight_soup_public_c4_${RUN_STAMP}.log
mkdir -p "$BASE_OUT" "$SOUP_ROOT" artifacts/logs

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 bash -lc '
set -euo pipefail
cd <repo-root>
RUN_STAMP=20260415T0531Z
BASE_OUT=artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms38_weight_soup_${RUN_STAMP}
SOUP_ROOT=artifacts/baselines/campp-ms38-weight-soup/${RUN_STAMP}
MS30=artifacts/baselines/campp-ms1-official-participants-lowlr/20260413T202123Z-d45cc7d9936e/campp_encoder.pt
MS31=artifacts/baselines/campp-ms1-official-participants-voxblink2-augment/20260413T203245Z-b87036ccb3db/campp_encoder.pt
MS32=artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt

run_candidate() {
  local run_id="$1"
  local out_dir="$BASE_OUT/$run_id"
  shift
  mkdir -p "$out_dir" "$SOUP_ROOT/$run_id"
  uv run --group train python scripts/build_campp_weight_soup.py \
    "$@" \
    --output-checkpoint "$SOUP_ROOT/$run_id/campp_encoder.pt" \
    --metadata-path "$SOUP_ROOT/$run_id/soup_metadata.json" \
    --reference-source MS32
  uv run --group train python scripts/run_official_campp_tail.py \
    --checkpoint-path "$SOUP_ROOT/$run_id/campp_encoder.pt" \
    --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
    --template-csv "datasets/Для участников/test_public.csv" \
    --data-root "datasets/Для участников" \
    --output-dir "$out_dir" \
    --experiment-id "$run_id" \
    --encoder-backend torch \
    --device cuda \
    --search-device cuda \
    --batch-size 512 \
    --search-batch-size 2048 \
    --top-cache-k 200 \
    --mode segment_mean \
    --eval-chunk-seconds 6.0 \
    --segment-count 3 \
    --long-file-threshold-seconds 6.0 \
    --frontend-pack-dir artifacts/cache/campp-official-public-ms1-v1-pack \
    --skip-save-embeddings \
    --skip-save-top-cache
  cp "$out_dir/submission_${run_id}_c4.csv" "$out_dir/submission.csv"
}

run_candidate MS38a_i065_campp_ms31_ms32_interp_${RUN_STAMP} \
  --source MS31="$MS31" --source MS32="$MS32" --weight MS31=0.35 --weight MS32=0.65
run_candidate MS38a_i075_campp_ms31_ms32_interp_${RUN_STAMP} \
  --source MS31="$MS31" --source MS32="$MS32" --weight MS31=0.25 --weight MS32=0.75
run_candidate MS38a_i085_campp_ms31_ms32_interp_${RUN_STAMP} \
  --source MS31="$MS31" --source MS32="$MS32" --weight MS31=0.15 --weight MS32=0.85
run_candidate MS38a_i095_campp_ms31_ms32_interp_${RUN_STAMP} \
  --source MS31="$MS31" --source MS32="$MS32" --weight MS31=0.05 --weight MS32=0.95
run_candidate MS38b_soup_ms30_ms31_ms32_w10_25_65_${RUN_STAMP} \
  --source MS30="$MS30" --source MS31="$MS31" --source MS32="$MS32" \
  --weight MS30=0.10 --weight MS31=0.25 --weight MS32=0.65
run_candidate MS38b_soup_ms30_ms31_ms32_w05_20_75_${RUN_STAMP} \
  --source MS30="$MS30" --source MS31="$MS31" --source MS32="$MS32" \
  --weight MS30=0.05 --weight MS31=0.20 --weight MS32=0.75
'
```

Status at launch:

- Synced `scripts/build_campp_weight_soup.py`,
  `tests/unit/test_campp_weight_soup.py`, and this history file to `remote`.
- Remote `ruff check` and remote targeted unit test passed inside `container`.
- Launched detached at `2026-04-15T05:31Z` with PID `504373`.
- Initial log shows first candidate build succeeded with `815` floating tensors averaged
  and `122` non-floating buffers copied from MS32; first public tail is running on GPU1.

Completion:

- Remote job completed successfully; PID `504373` is defunct and GPU1 returned to `0 MiB`.
- All six generated C4 submissions passed the local public validator.
- Aggregate report:
  `artifacts/reports/ms38/MS38_campp_weight_soup_public_c4_20260415T0531Z_summary.json`.
- Aggregate CSV:
  `artifacts/reports/ms38/MS38_campp_weight_soup_public_c4_20260415T0531Z_summary.csv`.

Results:

| Candidate | C4 top10 mean | Gini@10 | Max in-degree | Label used | Mean overlap vs MS32 C4 | Top1 equal vs MS32 | SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `MS38a_i095_campp_ms31_ms32_interp_20260415T0531Z` | `0.653450` | `0.333040` | `63` | `0.886694` | `9.2221/10` | `92.86%` | `68a03a885d69a720bc6cebc947ca2ab5001fc34f5eda183e4ada935b9b9da8e3` |
| `MS38a_i085_campp_ms31_ms32_interp_20260415T0531Z` | `0.648033` | `0.332951` | `50` | `0.882484` | `8.5277/10` | `84.61%` | `a2a47d24f383b9c7eea581c219522c698dcc606bbc67c0ecbdfd46632a32b09e` |
| `MS38b_soup_ms30_ms31_ms32_w05_20_75_20260415T0531Z` | `0.643902` | `0.333534` | `59` | `0.877362` | `8.0690/10` | `78.16%` | `532f4672c99cd282a57b191a86b3bc8249f76b88d95efbbc78248797f43d2735` |
| `MS38a_i075_campp_ms31_ms32_interp_20260415T0531Z` | `0.643178` | `0.333579` | `62` | `0.881942` | `8.0540/10` | `78.14%` | `09c729fa87a135d997eee6d932ae92badb0318d672cfb069e684ae927f4b7856` |
| `MS38b_soup_ms30_ms31_ms32_w10_25_65_20260415T0531Z` | `0.639894` | `0.334434` | `75` | `0.877547` | `7.6599/10` | `72.58%` | `dd33f2d408e20e07d9d1d8ad9c69b4be4b04fc0e4894130558dca3fcdabf9ae4` |
| `MS38a_i065_campp_ms31_ms32_interp_20260415T0531Z` | `0.638949` | `0.334229` | `59` | `0.876872` | `7.6561/10` | `72.68%` | `9496a6dabfb8c4e79946d714a3c72a9ba35129e6b85cd642b412915e8d8f8bff` |

Best candidate artifact:

- Checkpoint:
  `artifacts/baselines/campp-ms38-weight-soup/20260415T0531Z/MS38a_i095_campp_ms31_ms32_interp_20260415T0531Z/campp_encoder.pt`.
- Submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms38_weight_soup_20260415T0531Z/MS38a_i095_campp_ms31_ms32_interp_20260415T0531Z/submission.csv`.
- Local upload copy:
  `artifacts/submissions/MS38a_i095_campp_ms31_ms32_interp_20260415T0531Z_submission.csv`.
- Detailed summary:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms38_weight_soup_20260415T0531Z/MS38a_i095_campp_ms31_ms32_interp_20260415T0531Z/MS38a_i095_campp_ms31_ms32_interp_20260415T0531Z_summary.json`.

Public LB result:

- Submitted artifact:
  `artifacts/submissions/MS38a_i095_campp_ms31_ms32_interp_20260415T0531Z_submission.csv`.
- Validator status: passed locally before upload.
- SHA-256:
  `68a03a885d69a720bc6cebc947ca2ab5001fc34f5eda183e4ada935b9b9da8e3`.
- Public LB: `0.7396`.
- Delta: `+0.0017` absolute vs MS32 `0.7379`, `+0.6617` vs organizer baseline
  `0.0779`, and `-0.0077` vs MS41 `0.7473` if the MS41 row remains the active best.

Decision:

- MS31/MS32 interpolation behaves smoothly and monotonically toward MS32, and hidden public
  LB confirmed that the very near-MS32 point can improve over MS32 despite every tested
  interpolation being below the MS32 local public-graph proxy (`0.6535` best vs MS32
  `0.6564`).
- Three-way MS30/MS31/MS32 soups are weaker than near-MS32 interpolation, so they are
  rejected as replacement candidates.
- `MS38a_i095` is accepted as a confirmed improvement over MS32 and useful evidence that
  weight-space smoothing can help hidden boundary errors. It should not replace MS41 while
  MS41's `0.7473` public result is considered valid.
- CN-Celeb/SWA mode remains pending on the MS37 branch; no ready CN-Celeb-adapted CAM++
  checkpoint existed during this MS38 launch.
