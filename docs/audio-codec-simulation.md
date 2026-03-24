# Audio Codec Simulation

## Goal

Build one reproducible codec/channel simulation bank that later corruption, augmentation, and
stress-test stages can consume without guessing filter graphs, codec settings, or severity
coverage.

The authoritative source plan is
[`configs/corruption/codec-bank.toml`](../configs/corruption/codec-bank.toml). The reproducible
command is:

```bash
uv run python scripts/build_codec_bank.py
```

This renders deterministic preview artifacts into `artifacts/corruptions/codec-bank/` with:

- a synthetic source probe under `probe/`
- per-preset preview audio under `previews/`
- a machine-readable manifest at `manifests/codec_bank_manifest.jsonl`
- failure rows at `manifests/codec_bank_failures.jsonl`
- JSON and Markdown summaries under `reports/`

## Scope

This task covers preset definitions for the future codec/channel augmentation stages:

- band limiting
- telephony-style narrowband profiles
- low-rate VoIP-like profiles
- compression and quantization artifacts
- device/channel coloration and clipping

The current workflow deliberately uses a synthetic probe waveform instead of a committed audio
corpus. That keeps the smoke path deterministic and license-clean while still exercising the exact
FFmpeg transform stack.

## Preset Policy

The plan currently keeps five preset families:

- `band_limit`
- `telephony`
- `voip`
- `compression`
- `channel`

Each preset is tagged with a severity bucket:

- `light`: mostly bandwidth or EQ changes
- `medium`: audible telephony/VoIP degradation
- `heavy`: aggressive codec loss, clipping, or bit crushing

Severity weights in the plan are sampling hints for later augmentation policies. They do not imply
that every `heavy` preset should always be used in training.

## Reproducibility

The report records:

- the exact `ffmpeg` and `ffprobe` paths
- the detected FFmpeg version line
- the detected FFmpeg build configuration string
- per-preset command traces and preview SHA-256 hashes

This matters because codec availability and licensing details depend on the installed FFmpeg build.
The repository policy remains: log the build flags before release if GPL-only codecs or filters are
enabled.
