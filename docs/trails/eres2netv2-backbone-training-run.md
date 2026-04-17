# ERes2NetV2 Backbone Training Run

Date: 2026-04-11

Hypothesis: a from-scratch ERes2NetV2 trained on the participant speaker split should
produce stronger embeddings than the saturated organizer-style baseline encoder, and
must be evaluated through the existing C4 tail before any public submission.

Failed launch:

- Experiment id: `eres2netv2_participants_20260411_220149`
- Command/config: `configs/training/eres2netv2-participants-candidate.toml`, default
  `batch_size=64`, `eval_batch_size=64`, `device=cuda`.
- Log: `artifacts/logs/eres2netv2_participants_20260411_220149.log`
- Result: rejected/failed. CUDA OOM during the first training forward pass on RTX 4090;
  the process was stopped and no checkpoint or metric was produced.

Active launch:

- Experiment id: `eres2netv2_participants_b32_20260411_220247`
- Command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --group train python scripts/run_baseline.py \
  --model eres2netv2 \
  --config configs/training/eres2netv2-participants-candidate.toml \
  --project-override training.batch_size=32 \
  --project-override training.eval_batch_size=32 \
  --device cuda
```

- PID file: `artifacts/logs/eres2netv2_participants_b32_20260411_220247.pid`
- Log: `artifacts/logs/eres2netv2_participants_b32_20260411_220247.log`
- Status at launch: running on CUDA, GPU utilization reached `100%`, memory around
  `20.9 GiB`; no initial OOM observed.
- Decision: keep this as the first real ERes2NetV2 training attempt. After the checkpoint
  is written, evaluate it with `scripts/run_torch_checkpoint_c4_tail.py` on honest
  dense shifted v2 before any public leaderboard submission.
