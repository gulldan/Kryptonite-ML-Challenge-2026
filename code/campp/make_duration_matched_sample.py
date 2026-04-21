from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

V1_FEATURES = ["dur", "non_silent_ratio", "spectral_bandwidth_hz"]
V2_FEATURES = [
    "dur",
    "rms_dbfs",
    "non_silent_ratio",
    "leading_silence_sec",
    "trailing_silence_sec",
    "spectral_bandwidth_hz",
    "band_energy_ratio_3_8k",
]
V2_WEIGHTS = {
    "dur": 1.00,
    "rms_dbfs": 0.90,
    "non_silent_ratio": 0.80,
    "leading_silence_sec": 0.55,
    "trailing_silence_sec": 0.45,
    "spectral_bandwidth_hz": 0.70,
    "band_energy_ratio_3_8k": 0.50,
}
SUMMARY_FEATURES = [
    "dur",
    "rms_dbfs",
    "non_silent_ratio",
    "leading_silence_sec",
    "trailing_silence_sec",
    "spectral_bandwidth_hz",
    "band_energy_ratio_3_8k",
]
ACOUSTIC_SOURCE_COLUMNS = [
    "split",
    "filepath",
    "duration_sec",
    "rms_dbfs",
    "peak_dbfs",
    "crest_factor_db",
    "non_silent_ratio",
    "leading_silence_sec",
    "trailing_silence_sec",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff_hz",
    "spectral_flatness",
    "band_energy_ratio_0_1k",
    "band_energy_ratio_1_3k",
    "band_energy_ratio_3_8k",
    "zero_crossing_rate",
    "clipping_fraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build duration/acoustic-matched train subset for speaker retrieval."
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--acoustic-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=None,
        help="Optional manifest for comparison, e.g. v1 subset",
    )
    parser.add_argument("--sample-size", type=int, default=0, help="Default: len(test manifest)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration-bins", type=int, default=24)
    parser.add_argument("--strategy", choices=["v1", "v2"], default="v2")
    return parser.parse_args()


def _safe_qcut(values: pd.Series, q: int) -> tuple[pd.Series, np.ndarray]:
    _, edges = pd.qcut(values, q=q, retbins=True, duplicates="drop")
    edges = np.unique(edges.astype(float))
    if len(edges) < 2:
        edges = np.array([float(values.min()), float(values.max()) + 1e-6])
    labels = pd.cut(values, bins=edges, include_lowest=True, labels=False)
    return labels.astype(int), edges


def _manifest_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["ID", "dur", "path", "start", "stop", "spk", "orig_filepath"]
    return [column for column in preferred if column in df.columns]


def _normalize_manifest(df: pd.DataFrame, default_spk: str | None = None) -> pd.DataFrame:
    result = df.copy()
    if "path" not in result.columns and "filepath" in result.columns:
        result = result.rename(columns={"filepath": "path"})
    if "spk" not in result.columns:
        result["spk"] = default_spk or "unknown"
    if "ID" not in result.columns:
        result["ID"] = result["path"].astype(str).str.replace("/", "__", regex=False)
    if "start" not in result.columns:
        result["start"] = 0.0
    if "stop" not in result.columns:
        result["stop"] = np.nan
    if "orig_filepath" not in result.columns:
        result["orig_filepath"] = result["path"]
    return result


def load_features(
    train_manifest_path: Path, test_manifest_path: Path, acoustic_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_manifest = _normalize_manifest(pd.read_csv(train_manifest_path))
    test_manifest = _normalize_manifest(pd.read_csv(test_manifest_path), default_spk="test_public")
    acoustic = pd.read_parquet(acoustic_path, columns=ACOUSTIC_SOURCE_COLUMNS).copy()
    acoustic = acoustic.rename(columns={"filepath": "path", "duration_sec": "dur_acoustic"})
    feature_lookup = acoustic.drop_duplicates(subset=["path"])

    train_df = train_manifest.merge(
        feature_lookup.drop(columns=["split"]), on="path", how="left", validate="one_to_one"
    )
    test_df = test_manifest.merge(
        feature_lookup.drop(columns=["split"]), on="path", how="left", validate="one_to_one"
    )

    for df in (train_df, test_df):
        if "dur" in df.columns:
            df["dur"] = df["dur_acoustic"].fillna(df["dur"])
        else:
            df["dur"] = df["dur_acoustic"]

    missing_train = int(train_df[SUMMARY_FEATURES].isna().any(axis=1).sum())
    missing_test = int(test_df[SUMMARY_FEATURES].isna().any(axis=1).sum())
    if missing_train or missing_test:
        raise ValueError(
            f"Missing feature rows after merge: train={missing_train}, test={missing_test}"
        )
    return train_df, test_df


def _compute_match_scores(
    pool: pd.DataFrame,
    target_profile: pd.Series,
    scales: pd.Series,
    weights: dict[str, float],
    rng: np.random.Generator,
) -> pd.Series:
    score = np.zeros(len(pool), dtype=float)
    for feature, weight in weights.items():
        scale = max(float(scales.get(feature, 1.0)), 1e-6)
        score += float(weight) * ((pool[feature] - float(target_profile[feature])) / scale) ** 2
    score += rng.uniform(0.0, 1e-6, size=len(pool))
    return pd.Series(score, index=pool.index)


def _target_rows_by_bin(test: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    rows = test.groupby("duration_bin", observed=True).agg(target_rows=("ID", "size")).reset_index()
    rows["target_rows"] = (
        (rows["target_rows"] / rows["target_rows"].sum() * sample_size).round().astype(int)
    )
    diff = sample_size - int(rows["target_rows"].sum())
    if diff != 0:
        order = rows.sort_values("target_rows", ascending=(diff < 0)).index.tolist()
        idx_iter = iter(order * (abs(diff) + 2))
        while diff != 0:
            idx = next(idx_iter)
            new_value = rows.at[idx, "target_rows"] + (1 if diff > 0 else -1)
            if new_value >= 0:
                rows.at[idx, "target_rows"] = new_value
                diff += -1 if diff > 0 else 1
    return rows


def build_subset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_size: int,
    seed: int,
    duration_bins: int,
    strategy: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    train = train_df.copy()
    test = test_df.copy()

    active_features = V1_FEATURES if strategy == "v1" else V2_FEATURES
    active_weights = (
        {feature: 1.0 for feature in active_features}
        if strategy == "v1"
        else {feature: V2_WEIGHTS[feature] for feature in active_features}
    )

    test_duration_bin, edges = _safe_qcut(
        test["dur"], q=min(duration_bins, max(2, test["dur"].nunique()))
    )
    test["duration_bin"] = test_duration_bin
    train["duration_bin"] = pd.cut(train["dur"], bins=edges, include_lowest=True, labels=False)
    train = train.dropna(subset=["duration_bin"]).copy()
    train["duration_bin"] = train["duration_bin"].astype(int)

    target_rows = _target_rows_by_bin(test, sample_size)
    picks: list[pd.DataFrame] = []

    for row in target_rows.itertuples(index=False):
        pool = train.loc[train["duration_bin"] == row.duration_bin].copy()
        if pool.empty:
            continue
        bin_test = test.loc[test["duration_bin"] == row.duration_bin, active_features]
        target_profile = bin_test.median(numeric_only=True)
        scales = pd.concat([pool[active_features], bin_test], axis=0).std(ddof=0).replace(0, 1.0)
        pool["match_score"] = _compute_match_scores(
            pool, target_profile, scales, active_weights, rng
        )
        pool = pool.sort_values("match_score", kind="mergesort")
        picks.append(pool.head(min(int(row.target_rows), len(pool))))

    subset = (
        pd.concat(picks, ignore_index=False).drop_duplicates(subset=["ID"])
        if picks
        else train.head(0).copy()
    )
    if len(subset) < sample_size:
        used_ids = set(subset["ID"].tolist())
        remainder = train.loc[~train["ID"].isin(used_ids)].copy()
        global_target = test[active_features].median(numeric_only=True)
        global_scales = (
            pd.concat([train[active_features], test[active_features]], axis=0)
            .std(ddof=0)
            .replace(0, 1.0)
        )
        remainder["match_score"] = _compute_match_scores(
            remainder, global_target, global_scales, active_weights, rng
        )
        remainder = remainder.sort_values("match_score", kind="mergesort")
        subset = pd.concat([subset, remainder.head(sample_size - len(subset))], ignore_index=False)

    subset = subset.drop_duplicates(subset=["ID"]).head(sample_size).copy()
    if len(subset) != sample_size:
        raise ValueError(f"Subset size mismatch: expected {sample_size}, got {len(subset)}")
    return subset


def describe(df: pd.DataFrame, split_name: str) -> dict[str, float | int | str]:
    result: dict[str, float | int | str] = {
        "split": split_name,
        "rows": int(len(df)),
        "speakers": int(df["spk"].nunique()) if "spk" in df.columns else 0,
    }
    for column in SUMMARY_FEATURES:
        series = df[column]
        result[f"{column}_mean"] = float(series.mean())
        result[f"{column}_median"] = float(series.median())
        result[f"{column}_p10"] = float(series.quantile(0.10))
        result[f"{column}_p90"] = float(series.quantile(0.90))
    return result


def add_distance_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    test_row = summary_df.loc[summary_df["split"] == "test_public"].iloc[0]
    for metric in [
        "dur_median",
        "dur_mean",
        "rms_dbfs_median",
        "non_silent_ratio_median",
        "leading_silence_sec_median",
        "trailing_silence_sec_median",
        "spectral_bandwidth_hz_median",
        "band_energy_ratio_3_8k_median",
    ]:
        summary_df[f"delta_to_test__{metric}"] = (summary_df[metric] - test_row[metric]).abs()
    metric_cols = [c for c in summary_df.columns if c.startswith("delta_to_test__")]
    summary_df["delta_to_test__mean_selected_metrics"] = summary_df[metric_cols].mean(axis=1)
    return summary_df


def verify_outputs(output_dir: Path) -> list[str]:
    required = [
        output_dir / "train_manifest_duration_matched.csv",
        output_dir / "train_manifest_duration_matched.enriched.csv",
        output_dir / "stats_summary.csv",
        output_dir / "README.md",
        output_dir / "build_metadata.json",
    ]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    return missing


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_features(
        args.train_manifest, args.test_manifest, args.acoustic_parquet
    )
    baseline_df = None
    if args.baseline_manifest is not None and args.baseline_manifest.exists():
        baseline_df, _ = load_features(
            args.baseline_manifest, args.test_manifest, args.acoustic_parquet
        )

    sample_size = args.sample_size or len(test_df)
    subset = build_subset(
        train_df,
        test_df,
        sample_size=sample_size,
        seed=args.seed,
        duration_bins=args.duration_bins,
        strategy=args.strategy,
    )

    subset_manifest_path = args.output_dir / "train_manifest_duration_matched.csv"
    subset[_manifest_columns(train_df)].to_csv(subset_manifest_path, index=False)

    enriched_subset_path = args.output_dir / "train_manifest_duration_matched.enriched.csv"
    enriched_columns = _manifest_columns(train_df) + [
        column
        for column in ACOUSTIC_SOURCE_COLUMNS
        if column not in {"split", "filepath", "duration_sec"}
    ]
    enriched_columns = [column for column in enriched_columns if column in subset.columns]
    subset[enriched_columns].to_csv(enriched_subset_path, index=False)

    summary_rows = [describe(train_df, "train_full")]
    if baseline_df is not None:
        summary_rows.append(describe(baseline_df, "train_duration_matched_sample_v1"))
    summary_rows.extend(
        [
            describe(subset, f"train_duration_matched_sample_{args.strategy}"),
            describe(test_df, "test_public"),
        ]
    )
    summary_df = add_distance_metrics(pd.DataFrame(summary_rows))
    summary_path = args.output_dir / "stats_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    bin_edges = np.unique(np.quantile(test_df["dur"], np.linspace(0, 1, 9)))
    bin_tables = [
        pd.cut(train_df["dur"], bins=bin_edges, include_lowest=True)
        .value_counts(sort=False)
        .rename("train_full"),
        pd.cut(subset["dur"], bins=bin_edges, include_lowest=True)
        .value_counts(sort=False)
        .rename(f"train_duration_matched_sample_{args.strategy}"),
        pd.cut(test_df["dur"], bins=bin_edges, include_lowest=True)
        .value_counts(sort=False)
        .rename("test_public"),
    ]
    if baseline_df is not None:
        bin_tables.insert(
            1,
            pd.cut(baseline_df["dur"], bins=bin_edges, include_lowest=True)
            .value_counts(sort=False)
            .rename("train_duration_matched_sample_v1"),
        )
    pd.concat(bin_tables, axis=1).fillna(0).astype(int).reset_index().rename(
        columns={"index": "duration_bin"}
    ).to_csv(args.output_dir / "duration_bin_counts.csv", index=False)

    summary_by_split = summary_df.set_index("split")
    metadata = {
        "strategy": args.strategy,
        "train_manifest": str(args.train_manifest),
        "test_manifest": str(args.test_manifest),
        "baseline_manifest": str(args.baseline_manifest) if args.baseline_manifest else None,
        "acoustic_parquet": str(args.acoustic_parquet),
        "sample_size": int(sample_size),
        "seed": int(args.seed),
        "duration_bins": int(args.duration_bins),
        "feature_columns_used": V1_FEATURES if args.strategy == "v1" else V2_FEATURES,
        "feature_weights": {feature: 1.0 for feature in V1_FEATURES}
        if args.strategy == "v1"
        else V2_WEIGHTS,
        "available_acoustic_columns": [
            c for c in ACOUSTIC_SOURCE_COLUMNS if c not in {"split", "filepath"}
        ],
        "output_manifest": str(subset_manifest_path),
        "output_enriched_manifest": str(enriched_subset_path),
        "summary_csv": str(summary_path),
        "duration_delta_vs_test_v1": float(
            summary_by_split.loc["train_duration_matched_sample_v1", "delta_to_test__dur_median"]
        )
        if "train_duration_matched_sample_v1" in summary_by_split.index
        else None,
        "duration_delta_vs_test_v2": float(
            summary_by_split.loc[
                f"train_duration_matched_sample_{args.strategy}", "delta_to_test__dur_median"
            ]
        ),
        "mean_selected_delta_v1": float(
            summary_by_split.loc[
                "train_duration_matched_sample_v1", "delta_to_test__mean_selected_metrics"
            ]
        )
        if "train_duration_matched_sample_v1" in summary_by_split.index
        else None,
        "mean_selected_delta_v2": float(
            summary_by_split.loc[
                f"train_duration_matched_sample_{args.strategy}",
                "delta_to_test__mean_selected_metrics",
            ]
        ),
    }
    with (args.output_dir / "build_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    readme = f"""# Duration/acoustic-matched train sample ({args.strategy})

Manifest-only train subset for speaker retrieval, targeted to match `test_public`
rather than internal test.

- Source train manifest: `{args.train_manifest}`
- Source test manifest: `{args.test_manifest}`
- Acoustic source: `{args.acoustic_parquet}`
- Baseline manifest: `{args.baseline_manifest}`
- Output manifest: `train_manifest_duration_matched.csv`
- Rows: {sample_size}
- Strategy: `{args.strategy}`

## Features used for matching

{json.dumps(metadata["feature_weights"], ensure_ascii=False, indent=2)}

Notes:
- `rms_dbfs` represents amplitude/loudness more robustly than `peak_dbfs`, so
  `peak_dbfs` and `crest_factor_db` were left out of the matching objective.
- `leading_silence_sec` and `trailing_silence_sec` were used directly instead of
  only relying on `non_silent_ratio`.
- For spectral profile, `spectral_bandwidth_hz` and `band_energy_ratio_3_8k` were
  used to avoid a highly redundant stack of centroid/rolloff/ZCR features.
- Selection rule: first match the `test_public` duration-bin distribution, then
  rank train clips inside each bin by weighted standardized distance to the
  bin-level `test_public` acoustic medians.

See `stats_summary.csv` for comparison against full train, optional v1 baseline, and `test_public`.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")

    missing = verify_outputs(args.output_dir)
    if missing:
        raise FileNotFoundError(f"Missing or empty outputs: {missing}")

    print(
        json.dumps(
            {
                "subset_rows": len(subset),
                "subset_speakers": int(subset["spk"].nunique()),
                "strategy": args.strategy,
                "output_manifest": str(subset_manifest_path),
                "missing_outputs": missing,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
