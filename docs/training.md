# Training

Текущий рабочий training path и инструкция по подключению новых архитектур.

## Generic baseline pipeline

Все модели обучаются через единый пайплайн `run_speaker_baseline()`.
Каждая архитектура — тонкая обёртка: создаёт encoder и делегирует.

```text
TOML config -> config loader -> BaselineConfig
                                     |
                              model-specific wrapper:
                                build encoder (nn.Module)
                                     |
                              run_speaker_baseline()
                                     |
     data -> sampler -> train_epochs -> checkpoint
                                     |
     export_dev_embeddings -> score_trials -> verification_eval_report
                                     |
                              SpeakerBaselineRunArtifacts
                                (EER, minDCF, score gap, checkpoint, report)
```

Модельный контракт: вход `[batch, frames, 80]` -> выход `[batch, embedding_dim]`.

## Обучить модель

Единый скрипт:

```bash
uv run python scripts/run_baseline.py --model campp --config configs/training/campp-baseline.toml --device cuda
uv run python scripts/run_baseline.py --model eres2netv2 --config configs/training/eres2netv2-ffsvc2022-surrogate.toml --device cuda
```

Или модель-специфичные скрипты:

```bash
uv run python scripts/run_campp_baseline.py --config configs/training/campp-baseline.toml
uv run python scripts/run_eres2netv2_baseline.py --config configs/training/eres2netv2-ffsvc2022-surrogate.toml
```

Каждый запуск создаёт в output_root:
- `*_encoder.pt` — checkpoint
- `verification_eval_report.json` — EER, minDCF, score gap
- `*_baseline_report.md` — markdown отчёт
- `dev_embeddings.npz` — эмбеддинги dev set

## Сравнить модели

Обучил модель A, обучил модель B — получил два `verification_eval_report.json`.
Сравниваешь EER / minDCF / score gap.

Для формального сравнения семейств: `src/kryptonite/eval/final_family_decision.py`.

## Добавить новую архитектуру

Четыре файла:

1. **Encoder** — `src/kryptonite/models/<name>/model.py`:
   ```python
   @dataclass(frozen=True, slots=True)
   class MyModelConfig:
       feat_dim: int = 80
       embedding_size: int = 256

   class MyModelEncoder(nn.Module):
       def forward(self, features: torch.Tensor) -> torch.Tensor:
           # [batch, frames, feat_dim] -> [batch, embedding_size]
   ```

2. **Config loader** — `src/kryptonite/training/<name>/config.py` (~50 строк):
   - Dataclass `MyModelBaselineConfig` с полями `project, data, model, objective, optimization, provenance`
   - Функция `load_my_model_baseline_config()` использует `load_baseline_toml_sections()`

3. **Pipeline wrapper** — `src/kryptonite/training/<name>/pipeline.py` (~35 строк):
   ```python
   def run_my_model_baseline(config, *, config_path, device_override=None):
       device = resolve_device(device_override or config.project.runtime.device)
       encoder = MyModelEncoder(config.model).to(device)
       return run_speaker_baseline(
           config, encoder=encoder, embedding_size=config.model.embedding_size,
           model_config_dict=asdict(config.model), baseline_name="MyModel",
           report_file_name="my_model_baseline_report.md",
           embedding_source="my_model_baseline", tracker_kind="my-model-baseline",
           config_path=config_path, device=device,
       )
   ```

4. **TOML config** — `configs/training/my-model-baseline.toml`

Далее: зарегистрировать в `training/__init__.py` и `scripts/run_baseline.py`.

## Bring-up commands

```bash
uv sync --dev --group train --group tracking
uv run python scripts/training_env_smoke.py --config configs/deployment/train.toml
# GPU сервер:
uv run python scripts/training_env_smoke.py --config configs/deployment/train.toml --require-gpu
```

## Current model families

- **CAM++** — primary student path (baseline -> stage 2 -> stage 3 -> shortlist/selection)
- **ERes2NetV2** — comparison baseline

## Archive

Stage-specific и experiment-specific docs:
[archive/README.md](./archive/README.md)
