# Final Family Decision

`KRYP-058` freezes the next export-target family before any more ONNX/TensorRT
work starts.

## Decision

- Production student family: `CAM++`
- Stretch teacher branch: `WavLM / w2v-BERT` in `PEFT` mode
- Rejected export-first alternative: `ERes2NetV2`

## Why This Decision Exists

The checked-in release postmortem already makes the dependency chain explicit:

1. `KVA-536` freeze one family
2. `KVA-538` export that family to ONNX
3. `KVA-539` prove ONNX Runtime parity
4. `KVA-544` package the real fallback chain only after parity is real

Without a frozen family, the next cycle would duplicate export and deploy work
across incompatible model paths.

## Why CAM++ Wins The Export Slot

- The repository already contains the full staged CAM++ path:
  baseline, stage-2, stage-3, shortlist, and final-candidate selection
  contracts.
- That makes CAM++ the only repo-native student family that can move from
  current artifacts to a real export candidate without inventing missing
  selection tooling during the export cycle.
- The checked-in baseline evidence remains healthy enough for this purpose:
  `artifacts/baselines/campp/20260326T184437Z-2cf1cdc8af62/verification_eval_report.json`
  reports `EER=0.0`, `minDCF=0.0`, and a positive/negative score gap of
  `0.001648` on the tiny demo set.

The key point is not that CAM++ dominates every metric on the toy fixture. The
key point is that CAM++ is the only family with a mature repo-native handoff
from training to final-candidate selection.

## Why ERes2NetV2 Is Rejected For Now

- `ERes2NetV2` is a real repo-native baseline and its tiny demo-set score gap
  is actually larger than CAM++ in the checked-in smoke artifact.
- That is not enough to make it the next export target, because the repository
  currently stops at the baseline recipe for this family.
- There is no staged fine-tuning, shortlist, or final-candidate selection flow
  for `ERes2NetV2`, so choosing it now would increase scope before ONNX parity
  even exists.

In short: `ERes2NetV2` stays as a useful comparison baseline, not as the next
export-critical path.

## Why The Teacher Branch Stays Stretch-Only

- Teacher-style work still has upside for robustness and future distillation.
- `KVA-531` already defines the realistic constraint: `WavLM / w2v-BERT` only in
  `PEFT` mode, without pretending a full fine-tune fits the current hardware
  budget.
- The release postmortem also explicitly de-scopes teacher-heavy work from the
  next must-have milestone.

That means the teacher branch remains selected as the stretch path, but it must
not block the student export/parity sequence.

## Evidence

- CAM++ staged configs and docs:
  `configs/training/campp-stage2.toml`,
  `configs/training/campp-stage3.toml`,
  `configs/training/campp-stage3-sweep-shortlist.toml`,
  `configs/training/campp-stage3-model-selection.toml`,
  `docs/campp-stage2-training.md`,
  `docs/campp-stage3-training.md`,
  `docs/campp-hyperparameter-sweep-shortlist.md`,
  `docs/campp-model-selection.md`
- CAM++ baseline artifact:
  `artifacts/baselines/campp/20260326T184437Z-2cf1cdc8af62/`
- ERes2NetV2 baseline artifact:
  `artifacts/baselines/eres2netv2/20260326T184454Z-d0eba6dc018d/`
- Stretch-teacher and dependency framing:
  `docs/release-postmortem.md`,
  `configs/release/release-postmortem-v2.toml`

## Next Steps

- `KVA-538`: implement `PyTorch -> ONNX` export for the frozen `CAM++` family
- `KVA-539`: build `ONNX Runtime` parity against the current `PyTorch` path
- `KVA-544`: package the real fallback chain only after parity is proven

## Rebuild

```bash
uv run python scripts/build_final_family_decision.py \
  --config configs/release/final-family-decision.toml
```
