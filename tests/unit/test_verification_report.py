from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from kryptonite.eval import (
    VERIFICATION_CALIBRATION_CURVE_JSONL_NAME,
    VERIFICATION_DET_CURVE_JSONL_NAME,
    VERIFICATION_ERROR_ANALYSIS_JSON_NAME,
    VERIFICATION_ERROR_ANALYSIS_MARKDOWN_NAME,
    VERIFICATION_HISTOGRAM_JSON_NAME,
    VERIFICATION_REPORT_JSON_NAME,
    VERIFICATION_REPORT_MARKDOWN_NAME,
    VERIFICATION_ROC_CURVE_JSONL_NAME,
    VERIFICATION_SLICE_BREAKDOWN_JSONL_NAME,
    VERIFICATION_SLICE_DASHBOARD_HTML_NAME,
    build_verification_evaluation_report,
    render_verification_evaluation_markdown,
    write_verification_evaluation_report,
)


def test_build_verification_evaluation_report_produces_curves_and_slice_breakdown(
    tmp_path: Path,
) -> None:
    scores_path, trials_path, metadata_path = _write_eval_fixtures(tmp_path)
    score_rows = [
        {
            "left_id": "speaker_alpha:enroll",
            "right_id": "speaker_alpha:test",
            "label": 1,
            "score": 0.92,
        },
        {
            "left_id": "speaker_alpha:enroll",
            "right_id": "speaker_bravo:test",
            "label": 0,
            "score": 0.21,
        },
        {
            "left_id": "speaker_bravo:enroll",
            "right_id": "speaker_bravo:test",
            "label": 1,
            "score": 0.87,
        },
        {
            "left_id": "speaker_bravo:enroll",
            "right_id": "speaker_alpha:test",
            "label": 0,
            "score": 0.12,
        },
    ]
    trial_rows = [json.loads(line) for line in trials_path.read_text().splitlines() if line.strip()]
    metadata_rows = pl.read_parquet(metadata_path).to_dicts()

    report = build_verification_evaluation_report(
        score_rows,
        scores_path=scores_path,
        trials_path=trials_path,
        metadata_path=metadata_path,
        trial_rows=trial_rows,
        metadata_rows=metadata_rows,
        histogram_bins=4,
        calibration_bins=2,
    )

    assert report.summary.metrics.eer == 0.0
    assert report.summary.metrics.min_dcf == 0.0
    assert len(report.roc_points) >= 2
    assert len(report.det_points) == len(report.roc_points)
    assert len(report.histogram) == 4
    assert len(report.calibration_bins) == 2
    assert report.error_analysis is not None
    assert report.error_analysis.summary.false_accept_count == 0
    assert report.error_analysis.summary.false_reject_count == 0

    dataset_slice = next(
        row
        for row in report.slice_breakdown
        if row.field_name == "dataset" and row.field_value == "fixture"
    )
    assert dataset_slice.trial_count == 4
    assert dataset_slice.eer == 0.0
    assert dataset_slice.min_dcf == 0.0

    noise_slice = next(
        row
        for row in report.slice_breakdown
        if row.field_name == "noise_slice" and row.field_value == "stationary/light"
    )
    assert noise_slice.trial_count == 4
    assert noise_slice.positive_count == 2
    assert noise_slice.negative_count == 2


def test_write_verification_evaluation_report_writes_all_artifacts(tmp_path: Path) -> None:
    score_rows = [
        {"left_id": "utt-a", "right_id": "utt-b", "label": 1, "score": 0.8},
        {"left_id": "utt-a", "right_id": "utt-c", "label": 0, "score": 0.1},
    ]

    report = build_verification_evaluation_report(
        score_rows,
        scores_path=tmp_path / "scores.jsonl",
        histogram_bins=2,
        calibration_bins=2,
    )
    written = write_verification_evaluation_report(report, output_root=tmp_path / "artifacts")

    assert Path(written.report_json_path).name == VERIFICATION_REPORT_JSON_NAME
    assert Path(written.report_markdown_path).name == VERIFICATION_REPORT_MARKDOWN_NAME
    assert Path(written.slice_dashboard_path).name == VERIFICATION_SLICE_DASHBOARD_HTML_NAME
    assert Path(written.roc_curve_path).name == VERIFICATION_ROC_CURVE_JSONL_NAME
    assert Path(written.det_curve_path).name == VERIFICATION_DET_CURVE_JSONL_NAME
    assert Path(written.calibration_curve_path).name == VERIFICATION_CALIBRATION_CURVE_JSONL_NAME
    assert Path(written.histogram_path).name == VERIFICATION_HISTOGRAM_JSON_NAME
    assert Path(written.slice_breakdown_path).name == VERIFICATION_SLICE_BREAKDOWN_JSONL_NAME
    assert written.error_analysis_json_path is not None
    assert written.error_analysis_markdown_path is not None

    for path in (
        written.report_json_path,
        written.report_markdown_path,
        written.slice_dashboard_path,
        written.roc_curve_path,
        written.det_curve_path,
        written.calibration_curve_path,
        written.histogram_path,
        written.slice_breakdown_path,
        written.error_analysis_json_path,
        written.error_analysis_markdown_path,
    ):
        assert Path(path).is_file()

    html = Path(written.slice_dashboard_path).read_text()
    assert "Verification Slice Dashboard" in html
    assert "Duration" in html


def test_write_verification_evaluation_report_writes_error_analysis_when_identifiers_exist(
    tmp_path: Path,
) -> None:
    scores_path, trials_path, metadata_path = _write_eval_fixtures(tmp_path)
    score_rows = [
        {
            "left_id": "speaker_alpha:enroll",
            "right_id": "speaker_alpha:test",
            "label": 1,
            "score": 0.34,
        },
        {
            "left_id": "speaker_alpha:enroll",
            "right_id": "speaker_bravo:test",
            "label": 0,
            "score": 0.87,
        },
        {
            "left_id": "speaker_bravo:enroll",
            "right_id": "speaker_bravo:test",
            "label": 1,
            "score": 0.91,
        },
        {
            "left_id": "speaker_bravo:enroll",
            "right_id": "speaker_alpha:test",
            "label": 0,
            "score": 0.11,
        },
    ]
    trial_rows = [json.loads(line) for line in trials_path.read_text().splitlines() if line.strip()]
    metadata_rows = pl.read_parquet(metadata_path).to_dicts()

    report = build_verification_evaluation_report(
        score_rows,
        scores_path=scores_path,
        trials_path=trials_path,
        metadata_path=metadata_path,
        trial_rows=trial_rows,
        metadata_rows=metadata_rows,
        histogram_bins=4,
        calibration_bins=2,
    )
    written = write_verification_evaluation_report(report, output_root=tmp_path / "artifacts")

    assert report.error_analysis is not None
    assert report.error_analysis.summary.false_accept_count == 1
    assert report.error_analysis.summary.false_reject_count == 1
    assert written.error_analysis_json_path is not None
    assert written.error_analysis_markdown_path is not None
    assert Path(written.error_analysis_json_path).name == VERIFICATION_ERROR_ANALYSIS_JSON_NAME
    assert (
        Path(written.error_analysis_markdown_path).name == VERIFICATION_ERROR_ANALYSIS_MARKDOWN_NAME
    )
    assert Path(written.error_analysis_json_path).is_file()
    assert Path(written.error_analysis_markdown_path).is_file()

    payload = json.loads(Path(written.error_analysis_json_path).read_text())
    assert payload["summary"]["false_accept_count"] == 1
    assert payload["summary"]["false_reject_count"] == 1
    assert payload["speaker_confusions"][0]["speaker_a"] == "speaker_alpha"
    assert payload["speaker_confusions"][0]["speaker_b"] == "speaker_bravo"

    markdown = Path(written.error_analysis_markdown_path).read_text()
    assert "# Verification Error Analysis" in markdown
    assert "Priority Weak Spots" in markdown
    assert "speaker_alpha" in markdown


def test_render_verification_evaluation_markdown_includes_score_normalization_inputs(
    tmp_path: Path,
) -> None:
    report = build_verification_evaluation_report(
        [
            {"left_id": "utt-a", "right_id": "utt-b", "label": 1, "score": 0.8},
            {"left_id": "utt-a", "right_id": "utt-c", "label": 0, "score": 0.1},
        ],
        scores_path=tmp_path / "verification_scores_as_norm.jsonl",
        raw_scores_path=tmp_path / "dev_scores.jsonl",
        score_normalization="as-norm",
        score_normalization_summary_path=tmp_path / "verification_score_normalization_summary.json",
        embeddings_path=tmp_path / "dev_embeddings.npz",
        cohort_bank_path=tmp_path / "cohort-bank",
        histogram_bins=2,
        calibration_bins=2,
    )

    markdown = render_verification_evaluation_markdown(report)

    assert "- Raw scores: `" in markdown
    assert "- Score normalization: `as-norm`" in markdown
    assert "- Score-normalization summary: `" in markdown
    assert "- Embeddings: `" in markdown
    assert "- Cohort bank: `" in markdown


def _write_eval_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    scores_path = tmp_path / "scores.jsonl"
    scores_path.write_text("")

    trials_path = tmp_path / "trials.jsonl"
    trials_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "left_id": "speaker_alpha:enroll",
                        "right_id": "speaker_alpha:test",
                        "label": 1,
                    }
                ),
                json.dumps(
                    {
                        "left_id": "speaker_alpha:enroll",
                        "right_id": "speaker_bravo:test",
                        "label": 0,
                    }
                ),
                json.dumps(
                    {
                        "left_id": "speaker_bravo:enroll",
                        "right_id": "speaker_bravo:test",
                        "label": 1,
                    }
                ),
                json.dumps(
                    {
                        "left_id": "speaker_bravo:enroll",
                        "right_id": "speaker_alpha:test",
                        "label": 0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metadata_path = tmp_path / "metadata.parquet"
    pl.DataFrame(
        [
            {
                "trial_item_id": "speaker_alpha:enroll",
                "speaker_id": "speaker_alpha",
                "dataset": "fixture",
                "channel": "mono",
                "role": "enrollment",
                "duration_seconds": 0.8,
                "corruption_family": "noise",
                "corruption_severity": "light",
                "corruption_metadata": {"corruption_category": "stationary"},
            },
            {
                "trial_item_id": "speaker_alpha:test",
                "speaker_id": "speaker_alpha",
                "dataset": "fixture",
                "channel": "mono",
                "role": "test",
                "duration_seconds": 1.0,
                "corruption_family": "noise",
                "corruption_severity": "light",
                "corruption_metadata": {"corruption_category": "stationary"},
            },
            {
                "trial_item_id": "speaker_bravo:enroll",
                "speaker_id": "speaker_bravo",
                "dataset": "fixture",
                "channel": "phone",
                "role": "enrollment",
                "duration_seconds": 3.0,
                "corruption_family": "noise",
                "corruption_severity": "light",
                "corruption_metadata": {"corruption_category": "stationary"},
            },
            {
                "trial_item_id": "speaker_bravo:test",
                "speaker_id": "speaker_bravo",
                "dataset": "fixture",
                "channel": "phone",
                "role": "test",
                "duration_seconds": 4.0,
                "corruption_family": "noise",
                "corruption_severity": "light",
                "corruption_metadata": {"corruption_category": "stationary"},
            },
        ]
    ).write_parquet(metadata_path)
    return scores_path, trials_path, metadata_path
