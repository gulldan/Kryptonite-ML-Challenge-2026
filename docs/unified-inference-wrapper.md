# Unified Inference Wrapper

## Goal

Keep one raw-audio inference contract for:

- local Python callers;
- the thin HTTP adapter;
- deploy/demo smoke checks.

That contract now lives in `src/kryptonite/serve/inferencer.py` as `Inferencer`.

## What It Wraps

`Inferencer.from_config(...)` builds one runtime object that owns:

- runtime/backend preflight via `build_serve_runtime_report(...)`;
- deploy artifact checks via `build_infer_artifact_report(...)`;
- model bundle metadata loading;
- offline enrollment cache preload;
- runtime audio loading, post-processing, and chunk-aware embedding;
- shared cosine scoring through `ScoringService`.

The current raw-audio embedding implementation is `feature_statistics`:

- audio loading uses the shared normalization + VAD policy;
- features come from the shared Fbank extractor;
- chunk pooling follows the configured demo/eval chunking policy;
- frame pooling uses `mean` or `mean_std`.

## Local Usage

```python
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.serve import Inferencer

config = load_project_config(config_path=Path("configs/deployment/infer.toml"))
inferencer = Inferencer.from_config(config=config)

embed_payload = inferencer.embed_audio_paths(
    audio_paths=["artifacts/demo-subset/test/speaker_alpha-test_01.wav"]
)
verify_payload = inferencer.verify_audio_paths(
    enrollment_id="speaker_alpha",
    audio_paths=["artifacts/demo-subset/test/speaker_alpha-test_01.wav"],
)
```

## HTTP Usage

The thin HTTP adapter now delegates to the same `Inferencer` instance.

Example `POST /embed` request:

```json
{
  "audio_paths": ["artifacts/demo-subset/test/speaker_alpha-test_01.wav"],
  "stage": "demo"
}
```

Example `POST /benchmark` request:

```json
{
  "audio_paths": [
    "artifacts/demo-subset/test/speaker_alpha-test_01.wav",
    "artifacts/demo-subset/test/speaker_bravo-test_01.wav"
  ],
  "iterations": 3,
  "warmup_iterations": 1
}
```

## Current Limits

- The wrapper is ready for local/service parity, but not yet for final upload transport.
- The backend abstraction is explicit, but only `feature_statistics` is implemented today.
- Model bundle metadata is already loaded and surfaced, so a later ONNX/TensorRT backend can slot
  under the same `Inferencer` API without changing callers.
