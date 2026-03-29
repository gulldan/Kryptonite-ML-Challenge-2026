# EPIC-00 Research Baseline

`KVA-467` is the research closeout note for the first repository block:
problem framing, rules interpretation, verification protocol, architecture, and
the initial experiment matrix.

The repo already contains the individual child deliverables. This note exists to
make the block auditable as one entrypoint instead of forcing the next phase to
reverse-engineer the outcome from scattered docs and configs.

## Scope

`EPIC-00 - Постановка задачи и reverse engineering кейса` covers:

- `KVA-479 / KRYP-001`: competition rules matrix and external-data risk register
- `KVA-480 / KRYP-002`: primary model-task contract
- `KVA-481 / KRYP-003`: internal verification protocol
- `KVA-482 / KRYP-004`: system architecture v1
- `KVA-483 / KRYP-005`: experiment matrix v1

## Deliverables

- [docs/dataton-rules-matrix.md](/Users/mk/git/Kryptonite-ML-Challenge-2026/docs/dataton-rules-matrix.md)
  Competition-facing allow/deny/unknown matrix for data, checkpoints, teacher use, and augmentations.
- [docs/model-task-contract.md](/Users/mk/git/Kryptonite-ML-Challenge-2026/docs/model-task-contract.md)
  Repository ADR that fixes `speaker verification` as the primary task and keeps identification as a compatible mode on the same embedding/scoring contract.
- [docs/internal-verification-protocol.md](/Users/mk/git/Kryptonite-ML-Challenge-2026/docs/internal-verification-protocol.md)
  Auditable snapshot format for clean dev bundles plus production-like corrupted suites and their required slice fields.
- [docs/system-architecture-v1.md](/Users/mk/git/Kryptonite-ML-Challenge-2026/docs/system-architecture-v1.md)
  End-to-end pipeline, module ownership, interfaces, and export/serve boundary.
- [docs/experiment-matrix-v1.md](/Users/mk/git/Kryptonite-ML-Challenge-2026/docs/experiment-matrix-v1.md)
  Sequenced must-have versus stretch experiment plan with GPU-hour envelopes and scope control.

## What This Unlocks

- `EPIC-03` and `EPIC-04`: preprocessing plus corruption-bank work now build against one explicit audio contract and one verification protocol.
- `EPIC-05` and `EPIC-06`: baseline and training work now inherit a frozen task definition, evaluation contract, and first experiment order.
- `EPIC-07`, `EPIC-09`, and `EPIC-10`: backend/export/demo work now build on an explicit encoder boundary instead of an implicit notebook-era assumption.

## Validation

Fast builders for the doc-level artifacts:

```bash
uv run python scripts/build_dataton_rules_matrix.py
uv run python scripts/build_model_task_contract.py --config configs/base.toml
uv run python scripts/build_system_architecture.py --config configs/base.toml
uv run python scripts/build_experiment_matrix.py --config configs/training/experiment-matrix-v1.toml
```

Full data-backed verification protocol path:

```bash
uv run python scripts/prepare_ffsvc2022_surrogate.py
uv run python scripts/build_corrupted_dev_suites.py --config configs/base.toml --plan configs/corruption/corrupted-dev-suites.toml
uv run python scripts/build_verification_protocol.py --config configs/eval/verification-protocol.toml --require-complete
```

Repo-local smoke coverage for the verification-protocol completeness guard:

```bash
uv run pytest tests/unit/test_verification_protocol.py
```

## Remaining Risks

- The official Dataton evaluation criteria are still pending publication on `2026-04-11`, so the rules matrix remains a working interpretation instead of a final compliance decision.
- The full corrupted-dev protocol requires surrogate manifests that are intentionally not checked into git; local smoke should rely on tests, while full protocol rebuilds should run after surrogate-data preparation on the target machine.
- The checked-in runtime still uses `feature_statistics`, so `EPIC-00` freezes the system contract and experiment order, not final learned-model quality.
