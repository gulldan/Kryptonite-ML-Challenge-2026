# Kryptonite MS41 final evidence bundle

Documentary release bundle for the final submitted candidate
`MS41_ms32_classaware_c4_weak_20260415T0530Z`.

## Summary

- Bundle id: `ms41-final-evidence-v1`
- Mode: `recorded_candidate`
- Final candidate run id: `MS41_ms32_classaware_c4_weak_20260415T0530Z`
- Final candidate public LB: `0.7473`
- Staged source artifacts: `18`
- Staged docs/configs/records: `12 / 2 / 2`
- Recorded external runtime references: `6`

| Kind | Path | Type | Files | SHA256 |
| --- | --- | --- | --- | --- |
| repository_readme | docs/repository-readme.md | file | 1 | 3ceff2fbe71e9d61f68ebdc6c7cf4bccf5597b8084c9bda34c41605604ceb6d9 |
| solution_report | docs/challenge-solution-report.md | file | 1 | 857f97efc2c6c3d8d3270e717a230d82956165c99dbc3f3048df95ff64dda790 |
| experiment_history | docs/challenge-experiment-history.md | file | 1 | 1d2f55e3b920b6775e789a59745fda13f1590aa657b74d6413ecbd6c715274ff |
| model_card | docs/model-card.md | file | 1 | 5739da9d687707f9d41363822ece31de67b97ac7609b8202154b3fb9564f225a |
| release_runbook | docs/release-runbook.md | file | 1 | f9b70871574e6fa9b4cffb1e4d8176b2ed40ccc4386caa2902070d4d3f9d04f8 |
| current_public_artifact | docs/current-public-submission-artifact.md | file | 1 | 4531dfb69298068b6129841e71935153cc988928f175dd3e41390541fd301226 |
| trail | docs/trail_01_ms41.md | file | 1 | 30a5c511b2f2bb3be4cb586f1b779278df0d945a7ba0efff9c4b55333371b0ac |
| trail | docs/trail_02_ms38.md | file | 1 | e518ce8b514d6a87500bf5d15f52095a7f09a9e346b698a0081e797639960cfb |
| trail | docs/trail_03_ms40.md | file | 1 | 3c2cc3f68fd9aa1313de5212b79e3aa911a3e33ed1ba19e466feef262b6722b1 |
| trail | docs/trail_04_ms32.md | file | 1 | 8678b982fdf29c9c10091c485ef547e87e3f882504a9b63a6d0d641445f102df |
| trail | docs/trail_05_ms36b.md | file | 1 | 1d83d1d7f44ab706ebcb3c81f37737b6d329a9451de3612a4a87ec379adf4b4f |
| chart | docs/public-lb-score.svg | file | 1 | c064138e089eaf78df4881b048e592b92d0bea810418e9232b1ab2527f366423 |
| config | configs/ms41-private-full.toml | file | 1 | 22f6cf5f16bf97d9942f0cc2441a53f558d297dbb549c4554ec871671f960cee |
| config | configs/tensorrt-fp16-ms32.toml | file | 1 | c9f7c1e8de6009c5ba638c03e7ef9390a3e41315b008af75452e17c2a163f94c |
| staged_record | records/ms32_summary.json | file | 1 | 0367b556a764668c3bafb6b18b3ae700e696e2e2007f89ce2a835abe1a30d16a |
| staged_record | records/ms32_validation.json | file | 1 | 517cbd59e52809aeb7cdbae9f8e457ab3d376e79ed3897fcd9c29c1cac9e213d |
| recorded_runtime_refs | sources/runtime_external_references.json | file | 1 | 92d86d955037c32af0fa98447ca282033eba844b5a3b77c06d3efa43f00a4027 |
| recorded_runtime_note | sources/runtime_external_references.md | file | 1 | 48ad9b7fa7ec4b073aeb5ada93fe1aee5149f81c13fc19d85c0e42a4ecee4cf3 |

## Warnings

- This is a documentary candidate bundle, not a fully materialized runtime bundle.
- The current workspace does not stage the final `MS41` submission CSV, summary JSON,
  overlap JSON, classifier caches, or remote log.
- The `<remote-host>` <remote-shell> alias was not resolvable from the current environment, so the bundle
  cannot claim a fresh runtime sync.
- The only staged runtime JSON evidence in this workspace is the parent `MS32` summary
  and validator output under `records/`.

## Recorded final runtime references

- Submission CSV:
  `artifacts/submissions/MS41_ms32_classaware_c4_weak_20260415T0530Z_submission.csv`
- Submission SHA-256:
  `8b58013c3a710ef7e4c9f2fc5466ee9b2918d2ee271b5eaaa095b4976e194e84`
- Completion time:
  `2026-04-15T05:32:11Z`
- Detailed recorded runtime paths are listed in
  `sources/runtime_external_references.json`.

## Notes

- The jury-facing logic now lives in `docs/challenge-solution-report.md`, while
  `docs/challenge-experiment-history.md` remains the full lab ledger.
- `docs/current-public-submission-artifact.md` has been updated to point to the current
  `MS41` candidate instead of the old baseline artifact.
