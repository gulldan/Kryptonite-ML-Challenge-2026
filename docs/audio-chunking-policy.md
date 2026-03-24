# Audio Chunking Policy

## Goal

Provide one explicit utterance-chunking contract for `train`, `eval`, and
`demo` so downstream stages stop inventing incompatible crop lengths, overlap
rules, and aggregation behavior.

The current repository contract is:

- training keeps full utterances up to `4 s`
- training uses random crops in the `1-4 s` range only for longer utterances
- training short clips below `1 s` use an explicit policy instead of silent
  ad hoc padding; the default is `repeat_pad`
- eval and demo keep full utterances up to `4 s`
- eval and demo split longer audio into overlapping windows
- chunk-level outputs are pooled with an explicit strategy instead of implicit
  averaging in downstream code

## Config

The policy lives in `configs/base.toml`:

```toml
[chunking]
train_min_crop_seconds = 1.0
train_max_crop_seconds = 4.0
train_num_crops = 1
train_short_utterance_policy = "repeat_pad"
eval_max_full_utterance_seconds = 4.0
eval_chunk_seconds = 4.0
eval_chunk_overlap_seconds = 1.0
eval_pooling = "mean"
demo_max_full_utterance_seconds = 4.0
demo_chunk_seconds = 4.0
demo_chunk_overlap_seconds = 1.0
demo_pooling = "mean"
```

Supported `train_short_utterance_policy` values:

- `full`
- `repeat_pad`
- `zero_pad`

Supported pooling values:

- `mean`
- `max`

## Python API

Use the shared loader for waveform policy, then apply the shared chunking layer:

```python
from kryptonite.config import load_project_config
from kryptonite.data import AudioLoadRequest, load_audio
from kryptonite.features import (
    FbankExtractionRequest,
    chunk_utterance,
    extract_fbank,
    pool_chunk_tensors,
    UtteranceChunkingRequest,
)

config = load_project_config(config_path="configs/base.toml")
audio_request = AudioLoadRequest.from_config(config.normalization, vad=config.vad)
chunking_request = UtteranceChunkingRequest.from_config(config.chunking)
feature_request = FbankExtractionRequest.from_config(config.features)

loaded = load_audio(
    "datasets/ffsvc2022-surrogate/raw/dev/dev/ffsvc22_dev_000001.wav",
    project_root=config.paths.project_root,
    request=audio_request,
)
chunk_batch = chunk_utterance(
    loaded.waveform,
    sample_rate_hz=loaded.sample_rate_hz,
    stage="eval",
    request=chunking_request,
)

chunk_features = [
    extract_fbank(chunk.waveform, sample_rate_hz=loaded.sample_rate_hz, request=feature_request)
    for chunk in chunk_batch.chunks
]
chunk_embeddings = [encoder(features) for features in chunk_features]
pooled = pool_chunk_tensors(chunk_embeddings, pooling_mode=chunk_batch.pooling_mode)
```

## Stage Semantics

- `train`: full utterance for clips inside the medium range, random crops only
  for longer recordings, explicit handling for clips shorter than
  `train_min_crop_seconds`
- `eval`: full-utterance evaluation for short and medium clips, sliding windows
  for longer recordings, default pooling `mean`
- `demo`: same chunking primitive as eval, but with independent overlap and
  pooling knobs so the demo path can trade stability vs latency explicitly

When a long utterance does not align exactly with the hop size, the final window
is pinned to the end of the recording. That keeps full end coverage without
requiring downstream code to special-case a dropped tail.

## Current Limits

- chunking currently works on already loaded mono waveforms; it does not yet add
  a file-window iterator that avoids loading the full waveform before splitting
- pooling is intentionally limited to `mean` and `max`; any embedding-aware
  attention or score normalization stays in later tasks
- the task defines the policy and public API, not the future dataloader/batching
  layer that will eventually consume these chunks during training
