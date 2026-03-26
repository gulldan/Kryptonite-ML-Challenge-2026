"""Generate offline verification reports plus named threshold-calibration bundles."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kryptonite.eval import (
    DEFAULT_AS_NORM_STD_EPSILON,
    DEFAULT_AS_NORM_TOP_K,
    DEFAULT_SLICE_FIELDS,
    VERIFICATION_AS_NORM_SCORES_JSONL_NAME,
    VERIFICATION_SCORE_NORMALIZATION_SUMMARY_JSON_NAME,
    apply_as_norm_to_verification_scores,
    build_default_threshold_profile_specs,
    build_verification_evaluation_report,
    build_verification_threshold_calibration_report,
    load_verification_metadata_rows,
    load_verification_score_rows,
    load_verification_trial_rows,
    write_verification_evaluation_report,
    write_verification_threshold_calibration_report,
)

app = typer.Typer(add_completion=False, help=__doc__)

SCORES_OPTION = typer.Option(
    ...,
    "--scores",
    exists=True,
    dir_okay=False,
    readable=True,
    help="Path to a JSONL score file with `label` and `score` fields.",
)
TRIALS_OPTION = typer.Option(
    None,
    "--trials",
    exists=True,
    dir_okay=False,
    readable=True,
    help="Optional JSONL verification trials file used for richer metadata joins.",
)
METADATA_OPTION = typer.Option(
    None,
    "--metadata",
    exists=True,
    dir_okay=False,
    readable=True,
    help="Optional embedding metadata file (.jsonl or .parquet) used for slice breakdowns.",
)
EMBEDDINGS_OPTION = typer.Option(
    None,
    "--embeddings",
    exists=True,
    dir_okay=False,
    readable=True,
    help=(
        "Optional embedding matrix (.npz/.npy) required by score normalization methods "
        "like AS-norm."
    ),
)
COHORT_BANK_OPTION = typer.Option(
    None,
    "--cohort-bank",
    exists=True,
    file_okay=False,
    readable=True,
    help="Optional cohort-bank directory consumed by score normalization methods like AS-norm.",
)
OUTPUT_DIR_OPTION = typer.Option(
    None,
    "--output-dir",
    file_okay=False,
    help="Directory where report JSON/Markdown and calibration artifacts should be written.",
)
OUTPUT_OPTION = typer.Option(
    "text",
    "--output",
    help="Output format: text or json.",
    case_sensitive=False,
)
EVAL_SLICE_FIELDS_OPTION = typer.Option(
    list(DEFAULT_SLICE_FIELDS),
    "--slice-field",
    help="Repeatable slice field used for the verification evaluation package.",
)
THRESHOLD_SLICE_FIELDS_OPTION = typer.Option(
    None,
    "--threshold-slice-field",
    help="Repeatable slice field used for slice-aware threshold calibration.",
)
HISTOGRAM_BINS_OPTION = typer.Option(
    20,
    "--histogram-bins",
    help="Number of score histogram bins to emit.",
)
CALIBRATION_BINS_OPTION = typer.Option(
    10,
    "--calibration-bins",
    help="Number of calibration bins to emit.",
)
SCORE_NORMALIZATION_OPTION = typer.Option(
    "none",
    "--score-normalization",
    help="Score normalization to apply before metrics: none or as-norm.",
    case_sensitive=False,
)
AS_NORM_TOP_K_OPTION = typer.Option(
    DEFAULT_AS_NORM_TOP_K,
    "--as-norm-top-k",
    min=1,
    help="Top-k cohort scores to use per trial side when AS-norm is enabled.",
)
AS_NORM_STD_EPSILON_OPTION = typer.Option(
    DEFAULT_AS_NORM_STD_EPSILON,
    "--as-norm-std-epsilon",
    min=0.0,
    help="Positive floor applied to AS-norm cohort-score standard deviations.",
)
POINT_ID_FIELD_OPTION = typer.Option(
    "atlas_point_id",
    "--point-id-field",
    help="Metadata field aligned with the exported embedding point ids.",
)
EMBEDDINGS_KEY_OPTION = typer.Option(
    "embeddings",
    "--embeddings-key",
    help="Array key used when reading evaluation embeddings from a .npz artifact.",
)
IDS_KEY_OPTION = typer.Option(
    "point_ids",
    "--ids-key",
    help=(
        "Optional point-id array key used when reading evaluation embeddings from a .npz artifact."
    ),
)
P_TARGET_OPTION = typer.Option(0.01, "--p-target", help="Target prior used for minDCF.")
C_MISS_OPTION = typer.Option(1.0, "--c-miss", help="Miss cost used for minDCF.")
C_FA_OPTION = typer.Option(1.0, "--c-fa", help="False-accept cost used for minDCF.")
DEMO_TARGET_FAR_OPTION = typer.Option(
    0.05,
    "--demo-target-far",
    min=0.0,
    max=1.0,
    help="False-accept budget used for the demo operating point.",
)
PRODUCTION_TARGET_FAR_OPTION = typer.Option(
    0.01,
    "--production-target-far",
    min=0.0,
    max=1.0,
    help="False-accept budget used for the production operating point.",
)
MIN_SLICE_TRIALS_OPTION = typer.Option(
    25,
    "--min-slice-trials",
    min=1,
    help="Minimum number of slice trials required before emitting slice-aware thresholds.",
)


@app.command()
def main(
    scores: Path = SCORES_OPTION,
    trials: Path | None = TRIALS_OPTION,
    metadata: Path | None = METADATA_OPTION,
    embeddings: Path | None = EMBEDDINGS_OPTION,
    cohort_bank: Path | None = COHORT_BANK_OPTION,
    output_dir: Path | None = OUTPUT_DIR_OPTION,
    output: str = OUTPUT_OPTION,
    slice_field: list[str] | None = EVAL_SLICE_FIELDS_OPTION,
    threshold_slice_field: list[str] | None = THRESHOLD_SLICE_FIELDS_OPTION,
    histogram_bins: int = HISTOGRAM_BINS_OPTION,
    calibration_bins: int = CALIBRATION_BINS_OPTION,
    score_normalization: str = SCORE_NORMALIZATION_OPTION,
    as_norm_top_k: int = AS_NORM_TOP_K_OPTION,
    as_norm_std_epsilon: float = AS_NORM_STD_EPSILON_OPTION,
    point_id_field: str = POINT_ID_FIELD_OPTION,
    embeddings_key: str = EMBEDDINGS_KEY_OPTION,
    ids_key: str = IDS_KEY_OPTION,
    p_target: float = P_TARGET_OPTION,
    c_miss: float = C_MISS_OPTION,
    c_fa: float = C_FA_OPTION,
    demo_target_far: float = DEMO_TARGET_FAR_OPTION,
    production_target_far: float = PRODUCTION_TARGET_FAR_OPTION,
    min_slice_trials: int = MIN_SLICE_TRIALS_OPTION,
) -> None:
    resolved_output_dir = scores.parent if output_dir is None else output_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    score_rows = load_verification_score_rows(scores)
    resolved_scores_path: Path = scores
    raw_scores_path: Path | None = None
    score_normalization_summary_path: Path | None = None
    resolved_score_normalization = score_normalization.lower()
    if resolved_score_normalization == "as-norm":
        if metadata is None:
            raise typer.BadParameter("--metadata is required when --score-normalization=as-norm")
        if embeddings is None:
            raise typer.BadParameter("--embeddings is required when --score-normalization=as-norm")
        if cohort_bank is None:
            raise typer.BadParameter("--cohort-bank is required when --score-normalization=as-norm")

        as_norm_result = apply_as_norm_to_verification_scores(
            score_rows,
            embeddings_path=embeddings,
            metadata_path=metadata,
            cohort_bank_root=cohort_bank,
            top_k=as_norm_top_k,
            std_epsilon=as_norm_std_epsilon,
            embeddings_key=embeddings_key,
            ids_key=(None if not ids_key.strip() else ids_key),
            point_id_field=point_id_field,
        )
        raw_scores_path = scores
        resolved_scores_path = resolved_output_dir / VERIFICATION_AS_NORM_SCORES_JSONL_NAME
        score_normalization_summary_path = (
            resolved_output_dir / VERIFICATION_SCORE_NORMALIZATION_SUMMARY_JSON_NAME
        )
        resolved_scores_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in as_norm_result.score_rows),
            encoding="utf-8",
        )
        score_normalization_summary_path.write_text(
            json.dumps(as_norm_result.summary.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        score_rows = as_norm_result.score_rows
    elif resolved_score_normalization != "none":
        raise typer.BadParameter("score-normalization must be one of: none, as-norm")

    resolved_eval_slice_fields = tuple(slice_field)
    trial_rows = None if trials is None else load_verification_trial_rows(trials)
    metadata_rows = None if metadata is None else load_verification_metadata_rows(metadata)
    evaluation_report = build_verification_evaluation_report(
        score_rows,
        scores_path=resolved_scores_path,
        trials_path=trials,
        metadata_path=metadata,
        raw_scores_path=raw_scores_path,
        score_normalization=(
            None if resolved_score_normalization == "none" else resolved_score_normalization
        ),
        score_normalization_summary_path=score_normalization_summary_path,
        embeddings_path=embeddings,
        cohort_bank_path=cohort_bank,
        trial_rows=trial_rows,
        metadata_rows=metadata_rows,
        slice_fields=resolved_eval_slice_fields,
        histogram_bins=histogram_bins,
        calibration_bins=calibration_bins,
        p_target=p_target,
        c_miss=c_miss,
        c_fa=c_fa,
    )
    written_evaluation = write_verification_evaluation_report(
        evaluation_report,
        output_root=resolved_output_dir,
    )

    threshold_report = build_verification_threshold_calibration_report(
        score_rows,
        raw_score_rows=score_rows,
        trial_rows=trial_rows,
        metadata_rows=metadata_rows,
        profile_specs=build_default_threshold_profile_specs(
            demo_target_far=demo_target_far,
            production_target_far=production_target_far,
        ),
        slice_fields=tuple(threshold_slice_field or []),
        min_slice_trials=min_slice_trials,
        p_target=p_target,
        c_miss=c_miss,
        c_fa=c_fa,
    )
    written_thresholds = write_verification_threshold_calibration_report(
        threshold_report,
        output_root=resolved_output_dir,
    )

    if output == "json":
        typer.echo(
            json.dumps(
                {
                    "evaluation_report": evaluation_report.to_dict(include_curves=False),
                    "evaluation_outputs": written_evaluation.to_dict(),
                    "threshold_calibration": threshold_report.to_dict(),
                    "threshold_outputs": written_thresholds.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if output != "text":
        raise typer.BadParameter("output must be one of: text, json")

    metrics = evaluation_report.summary.metrics
    typer.echo(
        "\n".join(
            [
                "Verification threshold calibration complete",
                f"Scores: {scores}",
                *(
                    []
                    if raw_scores_path is None
                    else [f"Normalized scores: {resolved_scores_path}"]
                ),
                f"Trials: {metrics.trial_count}",
                f"Positives: {metrics.positive_count}",
                f"Negatives: {metrics.negative_count}",
                f"EER: {metrics.eer:.6f}",
                f"MinDCF@Ptarget={metrics.p_target:.4f}: {metrics.min_dcf:.6f}",
                f"Evaluation report: {written_evaluation.report_markdown_path}",
                f"Threshold report: {written_thresholds.report_markdown_path}",
                f"Slice-aware threshold groups: {threshold_report.summary.slice_group_count}",
            ]
        )
    )


if __name__ == "__main__":
    app()
