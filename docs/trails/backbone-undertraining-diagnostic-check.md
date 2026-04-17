# Backbone Undertraining Diagnostic Check

Date: 2026-04-12

Question:

- Are `P1_eres2netv2_h100_b128_public_c4` and `P2_campp_h100_b1024_public_c4`
  undertrained because both were trained for only `10` epochs?
- Do we log enough loss/accuracy data to answer this?

Artifacts:

- `artifacts/backbone_training_diagnostics/20260412T_undertraining_check/training_curves.csv`
- `artifacts/backbone_training_diagnostics/20260412T_undertraining_check/training_diagnostics_summary.json`
- Source logs copied from `remote`:
  - `artifacts/tracking/20260411T200748Z-15ced4a6d3ee/metrics.jsonl`
  - `artifacts/tracking/20260411T200858Z-757aa9406317/metrics.jsonl`

What is logged now:

- Per-epoch train loss, train accuracy, and learning rate are logged in
  `training_summary.json` and `artifacts/tracking/<run_id>/metrics.jsonl`.
- Final dev/retrieval metrics are logged after training: `eer`, `min_dcf`,
  `score_gap`, `rank_1_accuracy`, `rank_5_accuracy`, and `rank_10_accuracy`.
- Missing for a strong undertraining/overtraining decision: per-epoch dev metrics and
  per-epoch checkpoints. Current runs only validate the final epoch.

Observed curves:

| Model | Epoch 1 loss / acc | Epoch 10 loss / acc | Last-epoch loss delta | Last-epoch acc delta | Final LR | Final dev summary |
| --- | --- | --- | --- | --- | --- | --- |
| `ERes2NetV2` | `14.8727` / `0.2344` | `0.9641` / `0.9940` | `-0.1736` | `+0.0020` | `0.00005` | `rank1=0.9682`, `rank10=0.9888`, `eer=0.0300`, `score_gap=0.5689` |
| `CAM++` | `9.3825` / `0.3815` | `0.8162` / `0.9880` | `-0.0306` | `+0.0010` | `0.00005` | `rank1=0.9467`, `rank10=0.9803`, `eer=0.0443`, `score_gap=0.5222` |

Interpretation:

- The models are not grossly undertrained in the simple sense: train accuracy is already
  very high, and final dev metrics are strong.
- `ERes2NetV2` still reduces train loss at epoch 10, but the cosine scheduler has already
  decayed to `5e-5`; simply appending 4 epochs with the same ended schedule is unlikely
  to be a high-upside move.
- If testing longer training, use a new controlled recipe, not "continue blindly":
  `15-20` epochs from scratch or resume with an explicit low-LR fine-tune schedule, plus
  selected-epoch dev evaluation/checkpoints.
- Because public improved massively from backbone switch (`0.1249 -> 0.2410`) while
  local final dev is already high, the next likely bottleneck is not basic convergence.
  It is domain-shift-aware recipe and stronger/pretrained encoder work.

Decision:

- Do not spend the next slot on "same run plus 4 epochs" without extra validation.
- Add future logging requirement: any training-duration hypothesis must persist
  per-epoch or selected-epoch dev metrics and checkpoint paths, so the best epoch and
  overfit/underfit behavior can be recovered.
- A reasonable controlled follow-up is `ERes2NetV2` `15-20` epochs with a schedule designed
  for that length, plus per-epoch or every-2-epoch C4-tail dev evaluation.
