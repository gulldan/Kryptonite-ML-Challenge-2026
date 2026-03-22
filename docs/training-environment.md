# Training Environment

## Goal

Bring up a new machine with one `uv` command and verify that the project can import the training and experiment-tracking stack before any real training loop lands.

## Dependency Groups

- `infer`: `onnxruntime`
- `train`: `torch`, `torchaudio`, `onnx`, `onnxruntime`, `hydra-core`, `typer`
- `tracking`: `mlflow`, `wandb`

The repository keeps these groups explicit instead of stuffing them into the base runtime because:

- local repo tooling still works without the full ML stack when needed
- CPU/macOS machines and Linux GPU machines do not have identical binary constraints
- TensorRT uses NVIDIA-hosted wheel downloads that are not lock-friendly on non-Linux development hosts

## Bring-Up Commands

For a local development machine:

```bash
uv sync --dev --group train --group tracking
uv run python scripts/training_env_smoke.py
uv run python scripts/infer_smoke.py --config configs/deployment/infer.toml
```

For `gpu-server`:

```bash
uv sync --dev --group train --group tracking
uv pip install --python .venv/bin/python --extra-index-url https://pypi.nvidia.com/simple tensorrt-cu12
uv run python scripts/training_env_smoke.py --require-gpu
```

The smoke script fails fast if any required import is broken and prints the resolved versions plus lightweight backend details such as:

- `torch` CUDA or MPS availability
- `onnxruntime` execution providers
- `tensorrt` logger availability when the GPU stack is required

JSON output is available for automation:

```bash
uv run python scripts/training_env_smoke.py --output json
```

## Scope Decisions

- `torch` and `torchaudio` stay in the `train` group because they are required for baseline model and feature work.
- `onnxruntime` is also exposed as a standalone `infer` group so the runtime/demo container can stay smaller than the training image.
- `mlflow` and `wandb` are installed but not wired in as the default tracker yet; the project still defaults to the lightweight local backend described in `docs/tracking.md`.
- TensorRT stays outside the lockfile because NVIDIA's current wheel-stub packaging cannot be resolved cleanly from this macOS development host. The install command above keeps the GPU path explicit and reproducible for `gpu-server`.

## Limitations

- The current GPU path targets CUDA 12 via `tensorrt-cu12`. If the production GPU environment moves to another CUDA major, update the install command in this document before syncing.
- The smoke check validates imports and runtime metadata only. It does not yet benchmark kernels, export ONNX graphs, or validate TensorRT engine builds.
