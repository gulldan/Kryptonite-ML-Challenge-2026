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
uv run python scripts/generate_demo_artifacts.py --config configs/deployment/infer.toml
uv run python scripts/training_env_smoke.py --config configs/deployment/train.toml
uv run python scripts/infer_smoke.py --config configs/deployment/infer.toml
```

For `gpu-server`:

```bash
mkdir -p .local/bin .cache/uv
install -m 755 /home/qwerty/.local/bin/uv .local/bin/uv
install -m 755 /home/qwerty/.local/bin/uvx .local/bin/uvx
export PATH="$PWD/.local/bin:$PATH"
export UV_CACHE_DIR="$PWD/.cache/uv"
export XDG_CACHE_HOME="$PWD/.cache"
uv sync --dev --group train --group tracking
uv pip install --python .venv/bin/python --extra-index-url https://pypi.nvidia.com/simple tensorrt-cu12
uv run python scripts/training_env_smoke.py --config configs/deployment/train.toml --require-gpu
```

The smoke script fails fast if any required import is broken and prints the resolved versions plus lightweight backend details such as:

- `torch` CUDA or MPS availability
- `onnxruntime` execution providers
- `tensorrt` logger availability when the GPU stack is required

JSON output is available for automation:

```bash
uv run python scripts/training_env_smoke.py --config configs/deployment/train.toml --output json
```

When the machine is expected to have real datasets and manifests mounted or synced already, add `--require-artifacts` to turn the artifact checks from advisory into a hard gate.

## Scope Decisions

- `torch` and `torchaudio` stay in the `train` group because they are required for baseline model and feature work.
- `onnxruntime` is also exposed as a standalone `infer` group so the runtime/demo container can stay smaller than the training image.
- `mlflow` and `wandb` are installed but not wired in as the default tracker yet; the project still defaults to the lightweight local backend described in `docs/tracking.md`.
- TensorRT stays outside the lockfile because NVIDIA's current wheel-stub packaging cannot be resolved cleanly from this macOS development host. The install command above keeps the GPU path explicit and reproducible for `gpu-server`.
- On `gpu-server`, keep `uv`, `uvx`, and the `uv` cache on the same `/mnt/storage/Kryptonite-ML-Challenge-2026` volume as the repository checkout. This avoids filling the smaller home-disk while syncing the heavy PyTorch/NVIDIA stack.

## Limitations

- The current GPU path targets CUDA 12 via `tensorrt-cu12`. If the production GPU environment moves to another CUDA major, update the install command in this document before syncing.
- The smoke check validates imports and runtime metadata only. It does not yet benchmark kernels, export ONNX graphs, or validate TensorRT engine builds.
