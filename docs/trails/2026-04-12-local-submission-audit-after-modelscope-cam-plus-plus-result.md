# 2026-04-12 — Local Submission Audit After ModelScope CAM++ Result

Context:

- User reported `0.5695` for default ModelScope CAM++ VoxCeleb and asked to run local
  submission checks because the issue might be how `submission.csv` is built.

Local checks:

- Searched local workspace for ModelScope/CAM++ submission artifacts. No
  ModelScope-named submission artifact is present locally yet.
- Audited all local public submission CSVs under `artifacts/`:
  `34` true submission files found, `34/34` passed `validate_submission()` against
  `datasets/Для участников/test_public.csv`.
- Audit report:
  `artifacts/diagnostics/submission_audit/all_public_submission_validation.json`.
- Writer/validator smoke:
  - `write_submission()` writes `neighbours` as one CSV field, e.g.
    `test_public/000000.flac,"1,2,3,4,5,6,7,8,9,10"`.
  - Validator rejects duplicate and self-match rows.
  - Smoke report:
    `artifacts/diagnostics/submission_audit/writer_validator_smoke.json`.
- Local metric semantics:
  - On grouped `train.csv`, manual order-neighbor `P@10` equals baseline
    `precision_at_k_from_indices()` result:
    manual `0.9512352271056341`, metric `0.951235227105634`.
  - Parser roundtrip shape for a local submission subset: `[1000, 10]`.
  - Smoke report:
    `artifacts/diagnostics/submission_audit/local_metric_semantics_smoke.json`.
- Key command checks:
  - `uv run pytest tests/unit/test_eda.py::test_submission_validator_checks_paths_and_neighbours tests/unit/test_organizer_baseline_fixed.py::test_calc_metrics_validates_template_order_and_index_bounds tests/unit/test_embedding_scoring.py -q`
    passed: `6 passed`.
  - A broader pytest command including `tests/unit/test_submission_bundle.py` failed
    during collection because `kryptonite.serve` does not currently export
    `build_submission_bundle`; this is unrelated to public submission CSV creation.

Conclusion:

- Local evidence does not support a generic submission-format or off-by-one bug in our
  public CSV writer/validator/metric path.
- The ModelScope CAM++ result should be treated as a backbone quality/provenance signal.
  Next action is to bring the ModelScope submission/embeddings into the repo, validate
  the exact CSV locally, then use those embeddings for C4/cluster/fusion/pseudo-label
  runs.
