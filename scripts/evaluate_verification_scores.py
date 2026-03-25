"""Compute verification metrics and write a full evaluation report."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kryptonite.eval import (
    DEFAULT_SLICE_FIELDS,
    build_verification_evaluation_report,
    load_verification_metadata_rows,
    load_verification_score_rows,
    load_verification_trial_rows,
    write_verification_evaluation_report,
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
OUTPUT_DIR_OPTION = typer.Option(
    None,
    "--output-dir",
    file_okay=False,
    help="Directory where report JSON/Markdown and curve artifacts should be written.",
)
OUTPUT_OPTION = typer.Option(
    "text",
    "--output",
    help="Output format: text or json.",
    case_sensitive=False,
)
SLICE_FIELDS_OPTION = typer.Option(
    list(DEFAULT_SLICE_FIELDS),
    "--slice-field",
    help="Repeatable slice field used for per-slice breakdowns.",
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
P_TARGET_OPTION = typer.Option(0.01, "--p-target", help="Target prior used for minDCF.")
C_MISS_OPTION = typer.Option(1.0, "--c-miss", help="Miss cost used for minDCF.")
C_FA_OPTION = typer.Option(1.0, "--c-fa", help="False-accept cost used for minDCF.")


@app.command()
def main(
    scores: Path = SCORES_OPTION,
    trials: Path | None = TRIALS_OPTION,
    metadata: Path | None = METADATA_OPTION,
    output_dir: Path | None = OUTPUT_DIR_OPTION,
    output: str = OUTPUT_OPTION,
    slice_field: list[str] = SLICE_FIELDS_OPTION,
    histogram_bins: int = HISTOGRAM_BINS_OPTION,
    calibration_bins: int = CALIBRATION_BINS_OPTION,
    p_target: float = P_TARGET_OPTION,
    c_miss: float = C_MISS_OPTION,
    c_fa: float = C_FA_OPTION,
) -> None:
    score_rows = load_verification_score_rows(scores)
    trial_rows = None if trials is None else load_verification_trial_rows(trials)
    metadata_rows = None if metadata is None else load_verification_metadata_rows(metadata)
    report = build_verification_evaluation_report(
        score_rows,
        scores_path=scores,
        trials_path=trials,
        metadata_path=metadata,
        trial_rows=trial_rows,
        metadata_rows=metadata_rows,
        slice_fields=tuple(slice_field),
        histogram_bins=histogram_bins,
        calibration_bins=calibration_bins,
        p_target=p_target,
        c_miss=c_miss,
        c_fa=c_fa,
    )
    written = write_verification_evaluation_report(
        report,
        output_root=(scores.parent if output_dir is None else output_dir),
    )
    metrics = report.summary.metrics

    if output == "json":
        typer.echo(json.dumps(report.to_dict(include_curves=False), indent=2, sort_keys=True))
        return
    if output != "text":
        raise typer.BadParameter("output must be one of: text, json")

    typer.echo(
        "\n".join(
            [
                "Verification score evaluation complete",
                f"Scores: {scores}",
                f"Trials: {metrics.trial_count}",
                f"Positives: {metrics.positive_count}",
                f"Negatives: {metrics.negative_count}",
                f"EER: {metrics.eer:.6f}",
                f"EER threshold: {metrics.eer_threshold:.6f}",
                f"MinDCF@Ptarget={metrics.p_target:.4f}: {metrics.min_dcf:.6f}",
                f"MinDCF threshold: {metrics.min_dcf_threshold:.6f}",
                f"Report JSON: {written.report_json_path}",
                f"Report Markdown: {written.report_markdown_path}",
            ]
        )
    )


if __name__ == "__main__":
    app()
