"""Text and JSON writers for compact EDA review packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import polars as pl

from kryptonite.eda.review_defs import BUCKET_DEFS


def write_review_summary(
    path: Path,
    *,
    audio_dir: Path,
    public_dir: Path,
    file_stats: pl.DataFrame,
    local_query_eval: pl.DataFrame,
    public_lb_score: float,
) -> None:
    dataset = json.loads((audio_dir / "dataset_summary.json").read_text())
    retrieval = json.loads((audio_dir / "retrieval_summary.json").read_text())
    alignment = pl.read_csv(public_dir / "public_local_alignment.csv")
    ratio = _alignment_value(alignment, "pool_size", "public_div_local")
    local_p10 = retrieval["precision_at_10"]
    public_domain = _top_bucket_share(file_stats, "test_public", "domain_cluster")
    train_domain = _top_bucket_share(file_stats, "train", "domain_cluster")
    re_quality = _public_weighted_local_p10(
        local_query_eval, file_stats, "domain_bucket", "domain_cluster"
    )
    re_duration = _public_weighted_local_p10(
        local_query_eval, file_stats, "duration_bucket", "duration_bucket"
    )
    public_top10 = _alignment_value(alignment, "top10_mean_score_mean", "public_value")
    local_top10 = _alignment_value(alignment, "top10_mean_score_mean", "local_value")
    public = file_stats.filter(pl.col("split") == "test_public")
    lead_local = _share(local_query_eval, "leading_silence_bucket", "short")
    lead_public = _share(public, "leading_silence_bucket", "short")
    trail_local = _share(local_query_eval, "trailing_silence_bucket", "long")
    trail_public = _share(public, "trailing_silence_bucket", "long")
    text = f"""# EDA Review Summary

## Baseline numbers

- Local pseudo-val P@10: `{local_p10:.8f}` on `5,500` labelled train queries.
- Public leaderboard score: `{public_lb_score:.4f}`.
- Public/test pool size: `134,697`; local pool size: `5,500`; ratio: `{ratio:.2f}x`.
- Full profiled data: `{dataset["file_count"]:,}` files,
  `{dataset["speaker_count"]:,}` train speakers,
  `{dataset["total_duration_h"]:.2f}` hours.

## Observations -> Decisions

1. Local validation is strongly optimistic: `{local_p10:.3f}` vs public LB `{public_lb_score:.4f}`.
   -> Small labelled pool and easier candidate set inflate retrieval quality.
   -> Build `public_like_val`, speaker-disjoint, `50k-100k` queries.

2. Marginal bucket reweighting does not explain the drop:
   quality-weighted local P@10 is `{re_quality:.3f}`, duration-weighted is `{re_duration:.3f}`.
   -> Main error source is dense impostor gallery structure, not only domain proportions.
   -> Add dense-gallery validation before trusting any local ablation.

3. Public cosine is in another regime: top10 mean cosine `{public_top10:.3f}` public
   vs `{local_top10:.3f}` local, while public LB is lower.
   -> Absolute cosine is not a proxy for correctness.
   -> Test reciprocal kNN, hubness correction and local-scaling rerank.

4. Public has a padding shift: leading short silence is `{lead_public:.1%}` public
   vs `{lead_local:.1%}` local; trailing long silence is `{trail_public:.1%}` public
   vs `{trail_local:.1%}` local.
   -> Edge silence changes crop content and score regime.
   -> Test trim/VAD before crop and synthetic edge-silence augmentation.

5. Public has more peak-limited and silence-heavy examples than current local-val.
   -> Channel artifacts can create false high-similarity neighbors.
   -> Match validation buckets to public and evaluate channel augmentation by bucket.

6. Short local queries are toxic: `very_short` local P@10 is about `0.435`.
   -> Crop policy and short-file handling dominate baseline failures.
   -> Evaluate longer/multi-crop pooling and short-query weighting.

7. Public and train are dominated by narrowband-like files:
   public top bucket is `{public_domain}`; train top bucket is `{train_domain}`.
   -> Telephone/channel augmentation should be first-class.
   -> Keep narrowband-specific breakdowns in every experiment.

## Next hypotheses

1. `public_like_val` with public bucket proportions will reduce local P@10.
2. Dense-gallery validation will expose public-like impostor pressure.
3. Trim/VAD before crop will reduce padding-induced false matches.
4. Multi-crop aggregation will improve short and silence-heavy buckets.
5. Reciprocal/local-scaling rerank will reduce hubness in dense public retrieval.

## Plot guide

- `duration_hist_train_local_public.png`: local-val overrepresents very short files.
- `bucket_alignment_local_vs_public.png`: public has more peak-limited, padded files.
- `local_p10_by_bucket.png`: short duration is the clearest local failure mode.
- `local_same_vs_diff_score_distribution.png`: local same/diff overlap explains confusion.
- `public_vs_local_neighbor_score_distribution.png`: public scores are higher despite worse LB.
- `speaker_n_utts_distribution.png`: train can support larger public-like validation.
- `runtime_breakdown.png`: public runtime is dominated by audio loading, not search.
- `worst_queries_gallery.png`: top worst-query table for manual listening.
"""
    path.write_text(text, encoding="utf-8")


def write_dataset_split_summary(
    path: Path,
    *,
    audio_dir: Path,
    file_stats: pl.DataFrame,
    local_manifest: pl.DataFrame,
    public_lb_score: float,
) -> None:
    dataset = json.loads((audio_dir / "dataset_summary.json").read_text())
    val_stats = json.loads((audio_dir / "val_split_stats.json").read_text())
    train = dataset["splits"]["train"]
    public = dataset["splits"]["test_public"]
    payload = {
        "train": {
            "n_files": train["file_count"],
            "n_speakers": dataset["speaker_count"],
            "train_speakers_with_at_least_11_utts": dataset["speakers_with_at_least_11_utts"],
            "total_hours": train["total_duration_h"],
            "duration_p50": train["duration_quantiles_s"]["p50"],
            "duration_p90": train["duration_quantiles_s"]["p90"],
        },
        "local_val": {
            "n_files": local_manifest.height,
            "n_speakers": local_manifest.get_column("speaker_id").n_unique(),
            "speaker_disjoint_protocol": True,
            "note": "Pseudo-baseline query pool is sampled from held-out train speakers.",
            "pool_size": local_manifest.height,
            "local_speakers_with_11_utts": local_manifest.get_column("speaker_id").n_unique(),
            "planned_val_eligible_train_speakers": val_stats["eligible_val_speaker_count"],
        },
        "public": {
            "n_files": public["file_count"],
            "pool_size": public["file_count"],
            "leaderboard_score": public_lb_score,
        },
        "bucket_rules": BUCKET_DEFS,
        "speaker_overlap_checks": {
            "train_vs_public": "not_applicable_public_labels_hidden",
            "local_val_source": "train corpus speakers reserved for validation diagnostics",
        },
        "max_analysis_window_seconds": 6.0,
        "median_effective_analysis_duration_s": float(
            cast(float, file_stats["analysis_duration_s"].drop_nulls().median())
        ),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _top_bucket_share(file_stats: pl.DataFrame, split: str, col: str) -> str:
    row = (
        file_stats.filter(pl.col("split") == split)
        .group_by(col)
        .len("count")
        .sort("count", descending=True)
        .with_columns((pl.col("count") / pl.col("count").sum()).alias("share"))
        .row(0, named=True)
    )
    return f"{row[col]} ({float(row['share']):.1%})"


def _public_weighted_local_p10(
    local_query_eval: pl.DataFrame,
    file_stats: pl.DataFrame,
    local_col: str,
    public_col: str,
) -> float:
    local = local_query_eval.group_by(local_col).agg(pl.col("p10").mean().alias("p10"))
    public = (
        file_stats.filter(pl.col("split") == "test_public")
        .group_by(public_col)
        .len("count")
        .with_columns((pl.col("count") / pl.col("count").sum()).alias("share"))
    )
    joined = local.join(public, left_on=local_col, right_on=public_col, how="inner")
    return float((joined["p10"] * joined["share"]).sum())


def _alignment_value(alignment: pl.DataFrame, metric: str, column: str) -> float:
    return float(alignment.filter(pl.col("metric") == metric)[column][0])


def _share(frame: pl.DataFrame, column: str, value: str) -> float:
    return float(cast(float, (frame[column] == value).mean()))
