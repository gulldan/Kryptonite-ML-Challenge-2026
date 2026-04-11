"""Requested EDA data inventory for CSV packs."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from kryptonite.eda.csv_constants import NUMERIC_AUDIO_METRICS


def build_inventory(
    *,
    output_dir: Path,
    file_stats: pl.DataFrame | None,
    speaker_stats: pl.DataFrame | None,
    domain_clusters: pl.DataFrame | None,
    baseline_onnx: Path | None,
    artifact_root: Path,
) -> pl.DataFrame:
    has_file_stats = file_stats is not None
    has_audio_metrics = has_file_stats and _non_null_count(file_stats, "duration_s") > 0
    has_speakers = speaker_stats is not None
    has_domains = domain_clusters is not None
    has_baseline = baseline_onnx is not None and baseline_onnx.is_file()
    has_retrieval = (artifact_root / "embedding_eval.parquet").is_file()
    has_submission = (artifact_root / "submission_validation_report.json").is_file()

    sections = [
        (
            "0",
            "model_probe",
            "model_probe_metadata.csv",
            "baseline.onnx",
            "ready" if has_baseline else "missing_source",
            [
                "onnx_input_name",
                "onnx_input_shape",
                "onnx_output_names",
                "embedding_dim",
                "logit_dim",
                "expected_sample_rate_from_baseline_cli",
                "expected_chunk_seconds_from_baseline_cli",
                "mono_waveform_input",
            ],
        ),
        (
            "0",
            "model_probe",
            "model_probe_experiment_grid.csv",
            "baseline.onnx + selected val audio",
            "pending_embeddings",
            [
                "self_cosine_resampling",
                "self_cosine_trim_silence",
                "self_cosine_gain",
                "self_cosine_crop_1s_2s_4s_6s",
                "single_vs_multicrop",
                "raw_vs_l2_embedding",
                "config_to_local_p_at_10",
            ],
        ),
        (
            "1",
            "dataset_overview",
            "dataset_summary.csv",
            "train.csv/test_public.csv/file_stats",
            "ready" if has_file_stats else "missing_source",
            [
                "file_count",
                "speaker_count",
                "total_duration_hours",
                "duration_p10_p50_p90_p99",
                "sample_rate_distribution",
                "channel_distribution",
            ],
        ),
        (
            "1",
            "dataset_overview",
            "speaker_stats.csv",
            "file_stats",
            "ready" if has_speakers else "missing_source",
            [
                "files_per_speaker",
                "duration_per_speaker",
                "top_speakers_by_file_count",
                "speakers_with_1_2_3_plus_files",
                "cumulative_speaker_coverage",
            ],
        ),
        (
            "2",
            "audio_quality_domains",
            "file_stats.csv",
            "decoded audio",
            "ready" if has_audio_metrics else "ready_manifest_only",
            list(NUMERIC_AUDIO_METRICS),
        ),
        (
            "2",
            "audio_quality_domains",
            "domain_cluster_summary.csv",
            "file_stats",
            "ready" if has_domains else "missing_source",
            [
                "domain_bucket",
                "cluster_size",
                "mean_audio_stats",
                "cluster_examples",
                "audio_outliers",
            ],
        ),
        (
            "3",
            "speaker_structure_validation",
            "speaker_stats.csv",
            "file_stats",
            "ready" if has_speakers else "missing_source",
            [
                "n_utts_per_speaker",
                "total_duration_per_speaker",
                "mean_duration_per_speaker",
                "std_duration_per_speaker",
                "short_utt_frac",
                "bad_audio_frac",
            ],
        ),
        (
            "3",
            "speaker_structure_validation",
            "val_split_stats.csv",
            "speaker_stats",
            "ready" if has_speakers else "missing_source",
            [
                "train_speaker_count",
                "val_speaker_count",
                "val_file_count",
                "eligible_val_speakers_ge_11_utts",
                "train_val_duration_similarity",
            ],
        ),
        (
            "3",
            "speaker_cohesion",
            "speaker_cohesion.csv",
            "speaker embeddings",
            "pending_embeddings",
            [
                "mean_pairwise_cosine_within_speaker",
                "std_pairwise_cosine_within_speaker",
                "worst_cohesion_speakers",
                "two_cluster_speaker_suspicion",
            ],
        ),
        (
            "4",
            "qualitative_audio_audit",
            "qualitative_audit_queue.csv",
            "file_stats/domain_clusters",
            "ready" if has_audio_metrics else "ready_manifest_only",
            [
                "random_examples",
                "duration_bucket_examples",
                "domain_bucket_examples",
                "outlier_examples",
                "manual_tag",
                "manual_comment",
            ],
        ),
        (
            "5",
            "retrieval_baseline",
            "embedding_eval.csv",
            "precomputed embeddings",
            "ready" if has_retrieval else "missing_source",
            [
                "overall_local_precision_at_10",
                "top1_correct",
                "first_correct_rank",
                "intra_speaker_cosine",
                "inter_speaker_cosine",
                "p_at_10_by_duration_bucket",
                "p_at_10_by_domain_cluster",
                "p_at_10_by_silence_bucket",
                "p_at_10_by_narrowband_bucket",
                "query_top10_neighbors",
            ],
        ),
        (
            "6",
            "failure_analysis",
            "worst_queries.csv",
            "embedding_eval",
            "ready" if has_retrieval else "missing_source",
            [
                "100_worst_queries",
                "worst_speakers",
                "confused_speaker_pairs",
                "errors_by_bucket",
                "same_speaker_median_cos",
                "different_speaker_95p_cos",
                "margin",
            ],
        ),
        (
            "7",
            "inference_submission_qa",
            "runtime_profile.csv",
            "inference run logs",
            "missing_source",
            [
                "audio_loading_time",
                "preprocessing_time",
                "embedding_extraction_time",
                "index_build_time",
                "nearest_neighbor_search_time",
                "csv_writing_time",
                "files_per_sec",
                "real_time_factor",
                "gpu_memory",
                "estimated_full_run_time",
            ],
        ),
        (
            "7",
            "inference_submission_qa",
            "submission_validation_report.csv",
            "submission.csv + test_public.csv",
            "ready" if has_submission else "missing_source",
            [
                "row_count_matches_template",
                "filepath_matches_template",
                "row_order_matches_template",
                "neighbours_exactly_10",
                "integer_indices",
                "no_duplicates",
                "no_self_match",
                "no_empty_values",
                "no_nan",
            ],
        ),
        (
            "8",
            "experiment_tracker",
            "experiment_log.csv",
            "manual/automated experiment records",
            "ready_template",
            [
                "experiment_id",
                "encoder_checkpoint",
                "input_sr",
                "crop_policy",
                "n_crops",
                "trim_silence",
                "vad",
                "normalize_audio",
                "embedding_norm",
                "similarity_metric",
                "augmentation_set",
                "loss",
                "sampler",
                "val_p_at_10",
                "top1",
                "runtime",
                "hypothesis",
                "result",
                "conclusion",
                "notes",
            ],
        ),
    ]
    rows = [
        {
            "section_id": section_id,
            "section": section,
            "item": item,
            "output_csv": output_csv,
            "source": source,
            "status": status,
        }
        for section_id, section, output_csv, source, status, items in sections
        for item in items
    ]
    return pl.DataFrame(rows).with_columns(pl.lit(str(output_dir.resolve())).alias("csv_pack_dir"))


def _non_null_count(frame: pl.DataFrame | None, column: str) -> int:
    if frame is None or column not in frame.columns:
        return 0
    return int(frame.get_column(column).drop_nulls().len())
