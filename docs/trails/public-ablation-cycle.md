# Public Ablation Cycle

- Best submitted inference-only ablation before graph postprocess:
  `B8_trim_3crop_reciprocal_local_scaling`, public LB `0.1223`.
- B8 public gain vs `baseline_fixed_v1`: `+0.0199` absolute, about `+19.4%` relative.
- B8 public gain vs organizer baseline: `+0.0444` absolute, about `+57.0%` relative.
- Submitted public ranking: B0 `0.1024` < B2 `0.1098` < B4 `0.1150` < B7 `0.1206` < B8 `0.1223`.
- Spearman rank correlation on B0/B2/B4/B7, plus B8 for honest shifted v2:
  - smoke val: `1.0`
  - dense gallery val: `0.8`
  - dense shifted val: `1.0`
  - honest dense shifted v2: `0.9`
- Public B7 hubness @10 improved vs preprocessing-only runs, and B8 reduces hubness
  further:
  - B2 Gini@10 `0.5010`, max in-degree `149`
  - B4 Gini@10 `0.5139`, max in-degree `174`
  - B7 Gini@10 `0.4056`, max in-degree `87`
  - B8 Gini@10 `0.3385`, max in-degree `56`
- Honest `dense_shifted_v2` now uses one synthetic channel condition per file, shared
  by all crops. Under that stricter protocol B7 remains the best local selector:
  B7 `0.2331`, B8 `0.2320`, B9 `0.2323`, B10 `0.2308`. Public still prefers B8,
  so hubness reduction is a useful signal even when local P@10 is slightly lower.

Artifacts:

- `artifacts/eda/public_ablation_cycle.zip`
- `artifacts/eda/validation_cycle_package.zip`
- `artifacts/eda/baseline_fixed_dense_shifted_v2_honest.zip`
- `artifacts/eda/next_cycle_review_package.zip`
