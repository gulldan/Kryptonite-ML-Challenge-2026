# Audio Pipeline Reference

## Goal

This is the single reference note for the repo-wide audio contract.
It replaces the need to open separate top-level notes just to understand normalization, VAD,
Fbank extraction and chunking defaults.

## Canonical contract

- sample rate: `16 kHz`
- channels: `mono`
- output format: `PCM16 WAV`
- frontend: `80`-bin log-Mel / Fbank
- preprocessing is explicit and config-driven
- training, eval and runtime reuse the same loader/frontend family

## Processing order

1. decode / resample / channel fold-down
2. optional bounded loudness normalization
3. optional boundary-only VAD trimming
4. utterance chunking
5. Fbank extraction
6. chunk pooling and downstream embedding/scoring

## Runtime defaults

- loudness normalization: supported, disabled by default
- VAD mode: `none` by default
- training crop window: `1-4 s`
- eval/demo window: `4 s` chunks with overlap by default
- pooling: explicit, not implicit downstream behavior

## Corruption and robustness work

Corruption banks, silence augmentation and scheduler policy are now archive-level detail.
Use them when you are working on robustness experiments, not when you are onboarding.

Key archive docs:

- [../archive/audio-normalization.md](../archive/audio-normalization.md)
- [../archive/audio-vad-trimming.md](../archive/audio-vad-trimming.md)
- [../archive/audio-fbank-extraction.md](../archive/audio-fbank-extraction.md)
- [../archive/audio-chunking-policy.md](../archive/audio-chunking-policy.md)
- [../archive/audio-silence-augmentation.md](../archive/audio-silence-augmentation.md)
- [../archive/audio-augmentation-scheduler.md](../archive/audio-augmentation-scheduler.md)
- [../archive/audio-corrupted-dev-suites.md](../archive/audio-corrupted-dev-suites.md)

## Rule of use

If a behavior materially changes waveforms, chunk boundaries, frontend tensors or eval comparability,
it belongs in config and reusable code first, and in archive/reference docs second.
