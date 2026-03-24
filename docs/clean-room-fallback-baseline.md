# Clean-Room Fallback Baseline

## Goal

Define the speaker-verification fallback path that remains valid when challenge rules forbid any
external teacher model, pretrained checkpoint, or warm-start artifact.

## Chosen Fallback

The repository uses the repo-native CAM++ baseline as the clean-room fallback because it already:

- trains from local manifests and labels inside this repository
- reuses the shared audio loader, normalization, Fbank frontend, and chunking policy
- exports the same checkpoint, embedding, and score artifacts as the other baselines
- does not require teacher distillation or pretrained initialization

The canonical restricted config is:

- `configs/training/campp-ffsvc2022-restricted-rules.toml`

That config pins a `provenance.ruleset = "restricted-rules"` contract and the config loader now
rejects any restricted run that tries to declare teacher or pretrained resources.

## Run Commands

Local smoke path:

```bash
uv run python scripts/run_campp_baseline.py \
  --config configs/training/campp-baseline.toml
```

`gpu-server` restricted fallback path:

```bash
uv run python scripts/run_campp_baseline.py \
  --config configs/training/campp-ffsvc2022-restricted-rules.toml \
  --device cuda
```

## Artifact Contract

Restricted runs write under:

- `artifacts/baselines/campp/restricted-rules/<run-id>/`

The generated report and `training_summary.json` include:

- `provenance_ruleset`
- `provenance_initialization`

The report also prints the explicit provenance notes from the config, so the artifact is auditable
without inspecting the training code.

## Scope And Limits

- This is a fallback recipe, not a claim that CAM++ is the best final model.
- The dataset path still depends on whichever local manifests have been prepared on the machine.
- `ffsvc2022-surrogate` is a server-side engineering stand-in, not the final Dataton dataset.
- Mixed precision, larger sweeps, and more advanced training stages stay in later tasks.
