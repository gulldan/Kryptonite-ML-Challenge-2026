# Audio Loudness Normalization

## Goal

Сделать loudness normalization явной и воспроизводимой частью preprocessing
contract, а не неявной настройкой downstream feature extraction или one-off gain
staging.

Текущий контракт поддерживает bounded RMS normalization:

- `none`: не менять уровень сигнала
- `rms`: подтягивать clip к целевому `dBFS` только через глобальный gain
- ограничивать amplification через `max_loudness_gain_db`
- ограничивать attenuation через `max_loudness_attenuation_db`
- не искажать waveform: report отдельно проверяет, что loudness stage остаётся
  scale-only transform без деградации baseline waveform

## Config

Параметры живут в `configs/base.toml` внутри `[normalization]`:

```toml
[normalization]
target_sample_rate_hz = 16000
target_channels = 1
output_format = "wav"
output_pcm_bits_per_sample = 16
peak_headroom_db = 1.0
dc_offset_threshold = 0.01
clipped_sample_threshold = 0.999
loudness_mode = "none"
target_loudness_dbfs = -27.0
max_loudness_gain_db = 20.0
max_loudness_attenuation_db = 12.0
```

Supported values:

- `loudness_mode = "none"`
- `loudness_mode = "rms"`

По умолчанию loudness normalization выключен, чтобы baseline behavior не
менялся молча. Для ablation или preprocessing comparison используйте override:

```bash
uv run python scripts/show_config.py \
  --config configs/base.toml \
  --override normalization.loudness_mode=rms \
  --override normalization.target_loudness_dbfs=-27.0
```

## Loader Contract

`AudioLoadRequest.from_config(...)` теперь протаскивает loudness policy в shared
loader вместе с resample/mono/VAD policy. Порядок применения такой:

1. source windowing
2. channel fold-down
3. resampling
4. optional VAD / trimming
5. optional bounded loudness normalization

`LoadedAudio` теперь дополнительно возвращает:

- `loudness_mode`
- `loudness_target_dbfs`
- `loudness_applied`
- `loudness_gain_db`
- `loudness_gain_clamped`
- `loudness_peak_limited`
- `loudness_skip_reason`
- `pre_loudness_rms_dbfs`
- `post_loudness_rms_dbfs`
- `loudness_alignment_error`
- `loudness_degradation_check_passed`

## Comparison Report

Для честного `before/after` comparison используйте CLI:

```bash
uv run python scripts/loudness_normalization_report.py \
  --config configs/base.toml \
  --override normalization.loudness_mode=rms \
  --manifest artifacts/manifests/ffsvc2022-surrogate/dev_manifest.jsonl \
  --output-dir artifacts/eda/loudness-normalization
```

Артефакты:

- `artifacts/eda/loudness-normalization/loudness_normalization_report.json`
- `artifacts/eda/loudness-normalization/loudness_normalization_report.md`
- `artifacts/eda/loudness-normalization/loudness_normalization_rows.jsonl`

Guard в отчёте считается пройденным, если loudness stage не вносит shape
distortion после оптимального scale alignment.

## Manifest Rewrite Flow

`scripts/normalize_audio_dataset.py` и manifest-driven normalization теперь
используют тот же bounded loudness policy. В переписанные manifest rows
добавляется provenance:

- `source_rms_dbfs`
- `normalized_rms_dbfs`
- `normalization_loudness_mode`
- `normalization_loudness_applied`
- `normalization_loudness_gain_db`
- `normalization_loudness_gain_clamped`
- `normalization_loudness_peak_limited`
- `normalization_loudness_degradation_check_passed`

## Current Limits

- используется RMS/dBFS, а не perceptual LUFS
- loudness stage остаётся global gain, без compressor/limiter behavior
- loudness comparison guard валидирует waveform preservation, а не downstream SV
  metric quality; модельный baseline check появится позже вместе с feature/model
  pipeline
