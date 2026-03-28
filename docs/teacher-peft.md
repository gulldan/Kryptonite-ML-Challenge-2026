# Teacher PEFT

`KVA-531` turns the stretch teacher lane into a runnable repository-native
workflow instead of keeping it as planning-only text.

The goal is deliberately narrow:

- keep one realistic teacher branch around `WavLM` / `w2v-BERT`;
- enforce `PEFT`-only adaptation so the branch stays plausible on one `24 GB RTX 4090`;
- export the same artifact classes as the baseline student runs:
  checkpoint metadata, dev embeddings, trials, cosine scores, verification
  report, and reproducibility snapshot.

This is still a stretch branch. It must not block the `CAM++ -> export -> parity`
lane, but it is now concrete enough to support later `KVA-533` distillation work.

## What It Runs

The checked-in path lives in:

- `configs/training/teacher-peft.toml`
- `scripts/run_teacher_peft.py`
- `src/kryptonite/training/teacher_peft/`

The pipeline:

1. loads manifest-backed train/dev rows through the existing audio loader and
   chunking contract;
2. materializes raw waveform crops instead of repo-native `fbank` tensors;
3. loads one Hugging Face speech encoder via `AutoModel` and wraps it with
   `LoRA` adapters;
4. pools the final hidden state into one normalized speaker embedding;
5. trains a cosine classifier with the same `ArcMarginLoss` used in the compact
   student baselines;
6. exports dev embeddings and writes the same offline verification artifacts as
   the student lane.

## Default Config

The default checked-in config uses:

- `microsoft/wavlm-base-plus`
- `bf16`
- micro-batch `1`
- eval batch `2`
- gradient accumulation `8`
- gradient checkpointing enabled
- frozen feature encoder
- `LoRA` rank `16` with `all-linear` targeting

That is the conservative schedule intended for one `4090`.

To switch to the larger alternate family, override:

```bash
uv run python scripts/run_teacher_peft.py \
  --config configs/training/teacher-peft.toml \
  --project-override 'training.batch_size=1' \
  --project-override 'training.eval_batch_size=1' \
  --project-override 'training.max_epochs=1' \
  --project-override 'runtime.device="cuda"' \
  --project-override 'tracking.enabled=true' \
  --project-override 'tracking.run_name_prefix="teacher-w2vbert"' \
  --project-override 'chunking.train_min_crop_seconds=3.0' \
  --project-override 'chunking.train_max_crop_seconds=3.0'
```

And in the config file or an override set:

```toml
[model]
model_id = "facebook/w2v-bert-2.0"
```

Lower the effective batch schedule further if the larger encoder does not fit.

## Output Layout

Each run writes under `artifacts/baselines/teacher-peft/<run-id>/`:

- `teacher_peft/adapter/`:
  PEFT adapter weights and adapter config
- `teacher_peft/feature_extractor/`:
  serialized Hugging Face feature extractor
- `teacher_peft/heads.pt`:
  projection-head state, classifier state, and speaker index
- `teacher_peft/checkpoint_metadata.json`:
  model id, adapter settings, and masked config snapshot
- `dev_embeddings.npz`
- `dev_embedding_metadata.jsonl`
- `dev_embedding_metadata.parquet`
- `dev_trials.jsonl`
- `dev_scores.jsonl`
- `verification_report.json`
- `verification_report.md`
- `teacher_peft_report.md`
- `reproducibility_snapshot.json`

That layout is intentionally sufficient for a later student-distillation stage:
the branch has one frozen adapter checkpoint plus one matching embedding export.

## Command

```bash
uv sync --dev --group train --group tracking
uv run python scripts/training_env_smoke.py
uv run python scripts/run_teacher_peft.py \
  --config configs/training/teacher-peft.toml \
  --device cuda
```

Use a populated `.env` when gated Hugging Face access is needed. The loader reads
`HUGGINGFACE_HUB_TOKEN` through the existing project secret contract.

## Limits

- The branch is intentionally `PEFT`-only; full fine-tuning is out of scope.
- The embedding head uses masked mean pooling over the final hidden state. That
  keeps the first runnable version small and transparent, but it is not the only
  possible teacher head.
- Some Hugging Face speech backbones emit an informational warning when `PEFT`
  and gradient checkpointing are combined without a traditional input-embedding
  table. The checked-in tiny-`WavLM` sanity run completed successfully, but
  long GPU runs should still be validated on `gpu-server` before treating the
  branch as stable.
- This workflow is not wired into serving, export, or the must-have release
  path.
- The competition-facing legality of external checkpoints still depends on the
  rules matrix; keep this branch separate from the final submission candidate
  until policy is explicit.
