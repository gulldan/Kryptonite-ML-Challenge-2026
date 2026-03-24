# Audio Silence And Pause Augmentation

## Goal

Сделать silence/pause robustness явной частью corruption stack, а не
скрытой эвристикой в будущих training recipes.

Текущий модуль покрывает три отдельные операции:

- boundary silence padding через добавление тишины в начало и конец clip
- intra-utterance pause insertion через вставку коротких silent spans внутрь речи
- pause ratio perturbation через растяжение или сжатие уже найденных внутренних пауз

Все три операции живут в одном config block и выключены по умолчанию.

## Config

Параметры живут в `configs/base.toml` внутри `[silence_augmentation]`:

```toml
[silence_augmentation]
enabled = false
max_leading_padding_seconds = 0.0
max_trailing_padding_seconds = 0.0
max_inserted_pauses = 0
min_inserted_pause_seconds = 0.08
max_inserted_pause_seconds = 0.25
pause_ratio_min = 1.0
pause_ratio_max = 1.0
min_detected_pause_seconds = 0.08
max_perturbed_pause_seconds = 0.6
analysis_frame_ms = 20.0
silence_threshold_dbfs = -45.0
```

Практический смысл полей:

- `enabled`: главный kill-switch
- `max_leading_padding_seconds` / `max_trailing_padding_seconds`: сколько новой тишины можно
  добавить по краям
- `max_inserted_pauses`: максимум новых внутренних пауз на clip
- `min_inserted_pause_seconds` / `max_inserted_pause_seconds`: диапазон длины вставляемых пауз
- `pause_ratio_min` / `pause_ratio_max`: множитель для уже найденных внутренних пауз
- `min_detected_pause_seconds`: минимальная длина тишины, которую считаем паузой
- `max_perturbed_pause_seconds`: верхний предел роста уже найденной паузы
- `analysis_frame_ms` / `silence_threshold_dbfs`: параметры детектора silent frames

По умолчанию все effect knobs выставлены в no-op, так что baseline behavior
остаётся стабильным.

## Public API

Используйте helpers из `kryptonite.data`:

```python
import random

from kryptonite.config import load_project_config
from kryptonite.data import analyze_silence_profile, apply_silence_augmentation, load_audio

config = load_project_config(config_path="configs/base.toml")
loaded = load_audio(
    "datasets/ffsvc2022-surrogate/raw/dev/dev/ffsvc22_dev_000001.wav",
    project_root=config.paths.project_root,
)

before = analyze_silence_profile(
    loaded.waveform,
    sample_rate_hz=loaded.sample_rate_hz,
    config=config.silence_augmentation,
)
augmented, decision = apply_silence_augmentation(
    loaded.waveform,
    sample_rate_hz=loaded.sample_rate_hz,
    config=config.silence_augmentation,
    rng=random.Random(config.runtime.seed),
)
```

`decision` фиксирует, какие именно операции сработали:

- `leading_padding_seconds`
- `trailing_padding_seconds`
- `inserted_pause_count`
- `inserted_pause_total_seconds`
- `perturbed_pause_count`
- `stretched_pause_count`
- `compressed_pause_count`
- `skip_reason`

## Ablation Report

Для воспроизводимого `before/after` comparison используйте CLI:

```bash
uv run python scripts/silence_augmentation_report.py \
  --config configs/base.toml \
  --override silence_augmentation.enabled=true \
  --override silence_augmentation.max_leading_padding_seconds=0.15 \
  --override silence_augmentation.max_trailing_padding_seconds=0.20 \
  --override silence_augmentation.max_inserted_pauses=2 \
  --override silence_augmentation.pause_ratio_min=0.9 \
  --override silence_augmentation.pause_ratio_max=1.4 \
  --manifest artifacts/manifests/ffsvc2022-surrogate/dev_manifest.jsonl \
  --output-dir artifacts/eda/silence-augmentation
```

Артефакты:

- `artifacts/eda/silence-augmentation/silence_augmentation_report.json`
- `artifacts/eda/silence-augmentation/silence_augmentation_report.md`
- `artifacts/eda/silence-augmentation/silence_augmentation_rows.jsonl`

Отчёт показывает:

- как меняются `duration`, `silence_ratio` и `interior_pause_count`
- сколько строк реально изменилось
- сколько строк получили boundary padding, новые паузы и pause perturbation
- какие utterances дали самые большие сдвиги по тишине

Это и есть текущая ablation для KVA-509: waveform-level сравнение влияния
тишины и пауз на входной сигнал до появления общего augmentation scheduler и
model-level training/eval checks.

## Current Limits

- это standalone primitive, а не scheduler по эпохам; orchestration остаётся
  задачей следующего этапа
- вставка пауз ищет low-energy точки по frame RMS, а не phoneme/word boundary
- report измеряет waveform-level изменения, а не downstream SV quality
- новый padding всегда нулевой; для channel/noise ambience его нужно будет
  комбинировать с noise/channel augments на следующем этапе
