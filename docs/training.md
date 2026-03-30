# Training

## Pipeline

Все модели обучаются через единый `run_speaker_baseline()`.
Каждая архитектура -- тонкая обёртка: создаёт encoder и делегирует.

```
TOML config -> config loader -> BaselineConfig
                                     |
                              run_speaker_baseline()
                                     |
     data -> sampler -> train -> checkpoint -> embed -> score -> EER/minDCF
```

Контракт encoder: `[batch, frames, 80]` -> `[batch, embedding_dim]`.

## Запуск

```bash
# CAM++ baseline
uv run python scripts/run_baseline.py \
  --model campp \
  --config configs/training/campp-baseline.toml \
  --device cuda

# ERes2NetV2 baseline
uv run python scripts/run_baseline.py \
  --model eres2netv2 \
  --config configs/training/eres2netv2-baseline.toml \
  --device cuda
```

### Переопределение параметров

```bash
uv run python scripts/run_baseline.py \
  --model campp \
  --config configs/training/campp-baseline.toml \
  --device cuda \
  --project-override 'training.batch_size=32' \
  --project-override 'training.max_epochs=100' \
  --project-override 'training.precision="bf16"'
```

### Артефакты

Каждый запуск создаёт в `output_root`:

| Файл | Описание |
|------|----------|
| `*_encoder.pt` | Checkpoint модели |
| `dev_embeddings.npz` | Embeddings dev set |
| `verification_eval_report.json` | EER, minDCF, score gap |
| `*_baseline_report.md` | Markdown отчёт |

## Сравнить модели

Обучил модель A, обучил модель B -- сравниваешь `verification_eval_report.json`:
EER, minDCF, score gap.

## Добавить новую архитектуру

Три файла + конфиг + регистрация в скрипте.

### 1. Encoder -- `src/kryptonite/models/<name>/model.py`

```python
from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True, slots=True)
class MyModelConfig:
    feat_dim: int = 80
    embedding_size: int = 256

class MyModelEncoder(nn.Module):
    def __init__(self, config: MyModelConfig) -> None:
        super().__init__()
        # ... layers ...

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [batch, frames, 80]
        # return: [batch, embedding_size]
        ...
```

### 2. Config loader -- `src/kryptonite/training/<name>/config.py`

```python
from dataclasses import asdict, dataclass
from pathlib import Path
from kryptonite.config import ProjectConfig
from kryptonite.training.baseline_config import (
    BaselineDataConfig, BaselineObjectiveConfig,
    BaselineOptimizationConfig, BaselineProvenanceConfig,
)
from kryptonite.training.config_helpers import load_baseline_toml_sections

@dataclass(frozen=True, slots=True)
class MyModelBaselineConfig:
    base_config_path: str
    project_overrides: tuple[str, ...]
    project: ProjectConfig
    data: BaselineDataConfig
    model: MyModelConfig               # <-- your model config
    objective: BaselineObjectiveConfig
    optimization: BaselineOptimizationConfig
    provenance: BaselineProvenanceConfig

def load_my_model_baseline_config(
    *, config_path: Path | str, env_file=None, project_overrides=None,
) -> MyModelBaselineConfig:
    sections = load_baseline_toml_sections(
        config_path=config_path, env_file=env_file,
        project_overrides=project_overrides,
        data_defaults={
            "train_manifest": "artifacts/manifests/demo_manifest.jsonl",
            "dev_manifest": "artifacts/manifests/demo_manifest.jsonl",
            "output_root": "artifacts/baselines/my_model",
            "checkpoint_name": "my_model_encoder.pt",
        },
    )
    return MyModelBaselineConfig(
        base_config_path=sections.base_config_path,
        project_overrides=sections.project_overrides,
        project=sections.project,
        data=sections.data,
        model=MyModelConfig(**sections.model_section),
        objective=sections.objective,
        optimization=sections.optimization,
        provenance=sections.provenance,
    )
```

### 3. Pipeline wrapper -- `src/kryptonite/training/<name>/pipeline.py`

```python
from dataclasses import asdict
from pathlib import Path
from kryptonite.models.<name> import MyModelEncoder
from ..baseline_pipeline import run_speaker_baseline
from ..speaker_baseline import SpeakerBaselineRunArtifacts, resolve_device
from .config import MyModelBaselineConfig

def run_my_model_baseline(
    config: MyModelBaselineConfig, *, config_path: Path | str, device_override=None,
) -> SpeakerBaselineRunArtifacts:
    device = resolve_device(device_override or config.project.runtime.device)
    encoder = MyModelEncoder(config.model).to(device)
    return run_speaker_baseline(
        config, encoder=encoder,
        embedding_size=config.model.embedding_size,
        model_config_dict=asdict(config.model),
        baseline_name="MyModel",
        report_file_name="my_model_baseline_report.md",
        embedding_source="my_model_baseline",
        tracker_kind="my-model-baseline",
        config_path=config_path, device=device,
    )
```

### 4. TOML config -- `configs/training/my-model-baseline.toml`

```toml
base_config = "configs/base.toml"
project_overrides = [
  'runtime.num_workers=4',
  'training.precision="bf16"',
  'training.batch_size=16',
  'training.max_epochs=50',
  'chunking.train_min_crop_seconds=3.0',
  'chunking.train_max_crop_seconds=3.0',
  'chunking.train_num_crops=1',
]

[data]
train_manifest = "datasets/my_dataset/train.jsonl"
dev_manifest = "datasets/my_dataset/dev.jsonl"
output_root = "artifacts/baselines/my_model"
checkpoint_name = "my_model_encoder.pt"

[model]
embedding_size = 256

[objective]
scale = 32.0
margin = 0.2

[optimization]
optimizer_name = "adamw"
scheduler_name = "cosine"
learning_rate = 0.001
```

### 5. Регистрация в `scripts/run_baseline.py`

Добавить ветку в `main()`:

```python
elif normalized_model == "my_model":
    from kryptonite.training.<name> import load_my_model_baseline_config, run_my_model_baseline
    baseline = load_my_model_baseline_config(
        config_path=config, env_file=env_file, project_overrides=overrides
    )
    artifacts = run_my_model_baseline(baseline, config_path=config, device_override=device)
```

## Текущие модели

| Модель | Embedding dim | Config |
|--------|--------------|--------|
| CAM++ | 512 | `campp-baseline.toml` |
| ERes2NetV2 | 192 | `eres2netv2-baseline.toml` |
