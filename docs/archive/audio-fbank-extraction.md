# Audio Fbank Extraction

## Goal

Provide one explicit `80-dim` log-Mel/Fbank frontend for speaker work instead
of letting each downstream experiment silently pick different STFT, mel, CMVN,
or dtype defaults.

The current repository contract is:

- input waveform is already `16 kHz mono` from the shared loader/normalization path
- `80` mel bins
- `25 ms` frame length and `10 ms` frame shift
- `512`-point FFT with explicit Hann windowing
- right-padding on the last chunk so short utterances still produce features
- optional causal sliding CMVN for exact offline/online parity
- explicit output dtype casting after float32 computation

## Config

The feature frontend is configured in `configs/base.toml`:

```toml
[features]
sample_rate_hz = 16000
num_mel_bins = 80
frame_length_ms = 25.0
frame_shift_ms = 10.0
fft_size = 512
window_type = "hann"
f_min_hz = 20.0
power = 2.0
log_offset = 1e-6
pad_end = true
cmvn_mode = "none"
cmvn_window_frames = 300
output_dtype = "float32"
```

Supported `cmvn_mode` values:

- `none`
- `sliding`

Supported `output_dtype` values:

- `float32`
- `float16`
- `bfloat16`

`sliding` means causal windowed CMVN over already-emitted frames, not
utterance-global CMVN. That choice is deliberate so the streaming extractor can
match the offline path exactly without retroactively changing earlier frames.

## Python API

Use the loader for waveform policy, then the feature frontend for tensors:

```python
from kryptonite.config import load_project_config
from kryptonite.data import AudioLoadRequest, load_audio
from kryptonite.features import FbankExtractionRequest, extract_fbank

config = load_project_config(config_path="configs/base.toml")
audio_request = AudioLoadRequest.from_config(config.normalization, vad=config.vad)
feature_request = FbankExtractionRequest.from_config(config.features)

loaded = load_audio(
    "datasets/ffsvc2022-surrogate/raw/dev/dev/ffsvc22_dev_000001.wav",
    project_root=config.paths.project_root,
    request=audio_request,
)
features = extract_fbank(
    loaded.waveform,
    sample_rate_hz=loaded.sample_rate_hz,
    request=feature_request,
)
```

For chunked inference, reuse the same extractor parameters through the streaming
API:

```python
from kryptonite.features import FbankExtractor

extractor = FbankExtractor(request=feature_request)
online = extractor.create_online_extractor()

feature_chunks = []
for chunk in waveform_chunks:
    feature_chunks.append(online.push(chunk, sample_rate_hz=16_000))
feature_chunks.append(online.flush())
```

## Parity Smoke

Use the manifest-backed parity CLI to confirm offline and streaming extraction
stay aligned on a held-out split:

```bash
uv run python scripts/fbank_parity_report.py \
  --config configs/base.toml \
  --manifest artifacts/manifests/ffsvc2022-surrogate/dev_manifest.jsonl \
  --output-dir artifacts/eda/fbank-parity
```

The default run writes:

- `artifacts/eda/fbank-parity/fbank_parity_report.json`
- `artifacts/eda/fbank-parity/fbank_parity_report.md`
- `artifacts/eda/fbank-parity/fbank_parity_rows.jsonl`

The report uses a non-hop-aligned chunk size by default (`137 ms`) so parity is
checked on realistic streaming boundaries rather than on artificially perfect
frame-aligned splits.

## Current Limits

- only mono waveforms are accepted; multichannel handling stays in the loader
- the current frontend is intentionally explicit, not Kaldi-compatibility magic
- utterance-global CMVN is not implemented yet because it breaks exact streaming
  parity
- disk-backed feature caching and CPU/GPU profiling now live in
  [docs/audio-feature-cache.md](./audio-feature-cache.md)
