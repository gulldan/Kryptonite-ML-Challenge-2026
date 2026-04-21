# 2026-04-16 - SPEED2: NVIDIA ModelOpt + TensorRT maxopt5 overnight run

## Goal

Run the non-organizer model families through the most aggressive available
optimization path:

- NVIDIA ModelOpt ONNX AutoCast to FP16;
- TensorRT FP16 engine build with `builder_optimization_level=5`;
- full public `submission.csv` recompute;
- validator check and overlap comparison against the original source submit;
- TensorRT layer profile with engine footprint and ONNX parameter-size summary.

The organizer baseline is kept unchanged as the comparison reference. No
ModelOpt/TensorRT output is treated as score-equivalent unless it passes the
format validator and is compared against the original source CSV.

## Code and config changes

- `scripts/apply_modelopt_onnx_autocast.py`
  - wraps `modelopt.onnx.autocast.convert_to_mixed_precision`;
  - writes a patched model-bundle `metadata.json`;
  - keeps IO tensors in their original precision by default.
- `src/kryptonite/serve/tensorrt_engine_runtime.py`
  and `src/kryptonite/serve/tensorrt_generic.py`
  - expose TensorRT `builder_optimization_level`;
  - set detailed profiling verbosity when TensorRT exposes it.
- `scripts/profile_tensorrt_engine_layers.py`
  - records plan size, TensorRT device-memory footprint, tensor/layer counts,
    and optional ONNX initializer bytes grouped by first consuming node.
- `scripts/profile_campp_model_layers.py`
  - records parameter and buffer bytes per PyTorch module timing row.
- `configs/release/speed-family-modelopt-maxopt5.toml`
  - coordinates the three non-baseline family runs and source CSV comparisons.
- Model-specific maxopt5 build configs:
  - `configs/release/tensorrt-fp16-ms32-b128-segment6-modelopt-maxopt5.toml`;
  - `configs/release/tensorrt-fp16-w2vbert2-stage3-modelopt-maxopt5.toml`;
  - `configs/release/tensorrt-fp16-official-eres2net-h9-modelopt-maxopt5.toml`.

## Local validation

- `uv run ruff format` on touched Python files: passed.
- `uv run ruff check` on touched Python files: passed.
- `uvx ty check` on touched source and tests: passed.
- `uv run pytest tests/unit/test_tensorrt_engine.py -q`: `5 passed`.

## Remote package state

Remote target:

- host: `remote`;
- host repo: `<remote-repo>`;
- container: `container`;
- container repo: `<repo-root>`;
- GPU assignment: `CUDA_VISIBLE_DEVICES=1`.

Installed into the repo-local remote `.venv`:

```bash
uv pip install --python .venv/bin/python \
  --extra-index-url https://pypi.nvidia.com \
  'nvidia-modelopt[onnx]'
```

Verified versions in the container:

- `modelopt 0.43.0`;
- `onnx 1.21.0`;
- `tensorrt 10.16.1.11`.

## Remote launches

### Failed launch

- run id: `SPEED2_MODELOPT_MAXOPT5_20260416T202104Z`;
- log: `artifacts/logs/SPEED2_MODELOPT_MAXOPT5_20260416T202104Z.log`;
- result: failed before any model build or submit recompute;
- reason: shell heredoc quoting expanded helper arguments to empty strings, so
  the ModelOpt wrapper received the repository directory as the metadata path.

This is a launch-script failure only, not a model-quality result.

### Retry 1

- run id: `SPEED2_MODELOPT_MAXOPT5_20260416T202104Z_retry1`;
- script: `artifacts/logs/SPEED2_MODELOPT_MAXOPT5_20260416T202104Z_retry1.sh`;
- log: `artifacts/logs/SPEED2_MODELOPT_MAXOPT5_20260416T202104Z_retry1.log`;
- pid file:
  `artifacts/logs/SPEED2_MODELOPT_MAXOPT5_20260416T202104Z_retry1.pid`;
- latest pointers:
  - `artifacts/logs/latest_speed2_modelopt_maxopt5.log`;
  - `artifacts/logs/latest_speed2_modelopt_maxopt5.pid`.

Execution order:

1. CAM++ MS32 ModelOpt AutoCast, TensorRT maxopt5 build, layer profile, full
   public C4 submit, validator, and overlap-vs-MS41 comparison.
2. W2V1j stage3 ModelOpt AutoCast, generic TensorRT maxopt5 build, layer
   profile, full public C4 submit, validator, and overlap-vs-source comparison.
3. Official ERes2Net H9 ModelOpt AutoCast, generic TensorRT maxopt5 build,
   layer profile, full public C4 submit, validator, and overlap-vs-source
   comparison.

Current snapshot at launch monitoring:

- CAM++ ModelOpt AutoCast completed;
- ModelOpt converted `851/852` CAM++ ONNX nodes to FP16;
- the unsupported `Range` op stayed out of FP16 conversion;
- TensorRT maxopt5 CAM++ engine build was still running.

Final retry1 result:

- CAM++ completed end-to-end;
- W2V stopped before conversion because the ModelOpt wrapper only recognized
  `model_file` / `inference_package.artifacts.onnx_model_file`, while the
  W2V and ERes ONNX exporters wrote `model_path`.

Follow-up fix:

- `scripts/apply_modelopt_onnx_autocast.py` now accepts `model_path`;
- patched output metadata now writes both `model_file` and `model_path`;
- the ModelOpt import uses runtime `importlib` so local `ty` does not require
  `nvidia-modelopt` to be installed.
- validation after the fix:
  - local `ruff format`, `ruff check`, and `ty` passed for the wrapper;
  - remote `ruff check` and `ty` passed for the wrapper;
  - remote metadata smoke resolved both W2V and ERes source ONNX paths.

### Retry 2

- run id: `SPEED2_MODELOPT_MAXOPT5_20260416T202104Z_retry2_w2v_eres`;
- script:
  `artifacts/logs/SPEED2_MODELOPT_MAXOPT5_20260416T202104Z_retry2_w2v_eres.sh`;
- log:
  `artifacts/logs/SPEED2_MODELOPT_MAXOPT5_20260416T202104Z_retry2_w2v_eres.log`;
- result: completed W2V and ERes continuation.

ModelOpt conversion:

| Family | Converted nodes | Notes |
| --- | ---: | --- |
| CAM++ | `851/852` (`99.88%`) | `Range` stayed outside FP16. |
| W2V1j stage3 | `3452/3502` (`98.57%`) | `Range` and `Not` stayed outside FP16; overflow warning was emitted during cast. |
| ERes2Net H9 | `518/518` (`100.00%`) | Conversion completed cleanly apart from small-initializer warnings. |

## Final results

Full-submit timing excludes ONNX export, ModelOpt AutoCast, TensorRT build, and
layer profiling. It measures prepared public `submission.csv` generation only.

| Family | Full submit wall | Main stage timing | Validator | Source overlap mean@10 / top1 | Decision |
| --- | ---: | --- | --- | --- | --- |
| CAM++ MS32 ModelOpt maxopt5 + MS41 tail | `86.408s` (`88.771s` suite wall) | embedding/frontend `71.753s`, search `0.765s`, C4 rerank `11.415s`, write `0.628s` | passed | `8.585` / `87.31%` | Valid speed artifact, but slower than SPEED1 CAM++ `80.486s`; no reason to replace SPEED1. |
| W2V1j stage3 ModelOpt maxopt5 | `2494.929s` (`2497.533s` suite wall) | embedding/frontend `2491.926s`, search `0.786s`, rerank `1.271s` | passed | `0.0006` / `0.00%` | Rejected. TensorRT build validation produced `NaN`, and ranking overlap collapsed despite valid CSV format. |
| Official ERes2Net H9 ModelOpt maxopt5 | `344.473s` (`345.969s` suite wall) | embedding/frontend `331.352s`, search `0.818s`, C4 rerank `10.065s`, write `0.578s` | passed | `7.345` / `69.13%` | Valid speed artifact, but effectively tied with SPEED1 ERes `345.641s`; no meaningful win. |

TensorRT engine footprint:

| Family | Engine plan | TensorRT device memory | TensorRT layers | Total profiled layer mean |
| --- | ---: | ---: | ---: | ---: |
| CAM++ | `21.710 MiB` | `1012.8 MiB` | `1479` | `24.977ms` / batch `128` |
| W2V1j stage3 | `1139.170 MiB` | about `47.0 GiB` | `1561` | `2451.626ms` / batch `1024` |
| ERes2Net H9 | `58.957 MiB` | about `10.2 GiB` | `238` | `120.516ms` / batch `128` |

Top timed layer groups:

- CAM++: fused Conv/add/ReLU groups dominate; largest single profiled group
  `node_Conv_2537 + node_add_97 + node_relu_4` at `0.465ms`.
- W2V1j: repeated fused attention/softmax-like MYELIN groups dominate; top
  groups are each about `23.4ms`, spread across many transformer blocks.
- ERes2Net H9: early residual Conv/Add/ReLU groups dominate; top groups are
  about `3.5ms` each.

Top ONNX parameter footprints:

- CAM++: largest Conv initializers are about `1.0 MiB` each.
- W2V1j: feed-forward `MatMul` initializers are about `8.0 MiB` each.
- ERes2Net H9: final `/seg_1/Gemm` carries about `20.0 MiB`, followed by
  `/layer3_downsample/Conv` at `9.0 MiB`.

Conclusion:

- `builder_optimization_level=5` plus ModelOpt AutoCast did not improve the
  practical full-submit path over SPEED1.
- CAM++ remains the fastest recompute family, but the SPEED1 TensorRT artifact
  is faster than the ModelOpt maxopt5 artifact.
- W2V remains the best public-LB model family, but this aggressive FP16
  ModelOpt path is not score-safe.
- ERes remains an orthogonal diagnostic family; maxopt5 is valid but not faster.

## Output gates

For each optimized family, completion requires:

1. TensorRT build report and benchmark summary.
2. TensorRT layer-profile JSON and text log.
3. Full public submit generation timing.
4. `scripts/validate_submission.py` passes on the generated CSV.
5. `scripts/compare_submission_overlap.py` compares the generated CSV against
   the original source submit listed in
   `configs/release/speed-family-modelopt-maxopt5.toml`.
6. Final timings and layer/footprint summaries are copied back into this trail
   and the README speed section.

Until those gates pass, the optimized CSVs are speed diagnostics only.
