from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from kryptonite.eval import (
    VERIFICATION_THRESHOLD_CALIBRATION_JSON_NAME,
    VERIFICATION_THRESHOLD_CALIBRATION_MARKDOWN_NAME,
    ThresholdProfileSpec,
    build_verification_threshold_calibration_report,
    render_verification_threshold_calibration_markdown,
    write_verification_threshold_calibration_report,
)


def test_build_verification_threshold_calibration_report_selects_named_profiles() -> None:
    score_rows = [
        {"label": 1, "score": 0.92},
        {"label": 1, "score": 0.83},
        {"label": 1, "score": 0.62},
        {"label": 1, "score": 0.40},
        {"label": 0, "score": 0.71},
        {"label": 0, "score": 0.58},
        {"label": 0, "score": 0.32},
        {"label": 0, "score": 0.11},
    ]

    report = build_verification_threshold_calibration_report(
        score_rows,
        profile_specs=(
            ThresholdProfileSpec(name="balanced", selection_method="eer"),
            ThresholdProfileSpec(name="min_dcf", selection_method="min_dcf"),
            ThresholdProfileSpec(
                name="demo",
                selection_method="target_far",
                target_false_accept_rate=0.25,
            ),
            ThresholdProfileSpec(
                name="production",
                selection_method="target_far",
                target_false_accept_rate=0.0,
            ),
        ),
    )

    assert [profile.name for profile in report.global_profiles] == [
        "balanced",
        "min_dcf",
        "demo",
        "production",
    ]
    assert report.summary.global_profile_count == 4

    balanced = report.global_profiles[0]
    assert balanced.threshold == 0.62
    assert balanced.false_accept_rate == 0.25
    assert balanced.false_reject_rate == 0.25

    demo = report.global_profiles[2]
    assert demo.threshold == 0.62
    assert demo.target_false_accept_rate == 0.25
    assert demo.false_accept_rate == 0.25
    assert demo.false_reject_rate == 0.25

    production = report.global_profiles[3]
    assert production.threshold == 0.83
    assert production.target_false_accept_rate == 0.0
    assert production.false_accept_rate == 0.0
    assert production.false_reject_rate == 0.5

    min_dcf = report.global_profiles[1]
    assert min_dcf.selection_method == "min_dcf"
    assert min_dcf.p_target == 0.01
    assert min_dcf.c_miss == 1.0
    assert min_dcf.c_fa == 1.0


def test_threshold_calibration_report_emits_slice_aware_profiles_and_files(tmp_path: Path) -> None:
    trial_rows, metadata_path = _write_slice_fixture_metadata(tmp_path)
    score_rows = [
        {
            "left_id": "mono_alpha:enroll",
            "right_id": "mono_alpha:test",
            "label": 1,
            "score": 0.91,
        },
        {
            "left_id": "mono_alpha:enroll",
            "right_id": "mono_bravo:test",
            "label": 0,
            "score": 0.21,
        },
        {
            "left_id": "mono_bravo:enroll",
            "right_id": "mono_bravo:test",
            "label": 1,
            "score": 0.88,
        },
        {
            "left_id": "mono_bravo:enroll",
            "right_id": "mono_alpha:test",
            "label": 0,
            "score": 0.19,
        },
        {
            "left_id": "phone_charlie:enroll",
            "right_id": "phone_charlie:test",
            "label": 1,
            "score": 0.79,
        },
        {
            "left_id": "phone_charlie:enroll",
            "right_id": "phone_delta:test",
            "label": 0,
            "score": 0.47,
        },
        {
            "left_id": "phone_delta:enroll",
            "right_id": "phone_delta:test",
            "label": 1,
            "score": 0.74,
        },
        {
            "left_id": "phone_delta:enroll",
            "right_id": "phone_charlie:test",
            "label": 0,
            "score": 0.41,
        },
    ]

    report = build_verification_threshold_calibration_report(
        score_rows,
        raw_score_rows=score_rows,
        trial_rows=trial_rows,
        metadata_rows=pl.read_parquet(metadata_path).to_dicts(),
        profile_specs=(
            ThresholdProfileSpec(name="balanced", selection_method="eer"),
            ThresholdProfileSpec(
                name="demo",
                selection_method="target_far",
                target_false_accept_rate=0.5,
            ),
        ),
        slice_fields=("channel",),
        min_slice_trials=4,
    )

    assert report.summary.slice_group_count == 2
    slice_values = {(profile.slice_field, profile.slice_value) for profile in report.slice_profiles}
    assert slice_values == {("channel", "mono"), ("channel", "phone")}

    written = write_verification_threshold_calibration_report(
        report,
        output_root=tmp_path / "artifacts",
    )

    assert Path(written.report_json_path).name == VERIFICATION_THRESHOLD_CALIBRATION_JSON_NAME
    assert (
        Path(written.report_markdown_path).name == VERIFICATION_THRESHOLD_CALIBRATION_MARKDOWN_NAME
    )

    payload = json.loads(Path(written.report_json_path).read_text())
    assert payload["summary"]["slice_group_count"] == 2
    assert payload["slice_profiles"][0]["profiles"][0]["name"] == "balanced"

    markdown = render_verification_threshold_calibration_markdown(report)
    assert "# Verification Threshold Calibration" in markdown
    assert "Global Operating Points" in markdown
    assert "Slice-Aware Thresholds" in markdown
    assert "`mono`" in markdown
    assert "`phone`" in markdown


def _write_slice_fixture_metadata(tmp_path: Path) -> tuple[list[dict[str, str | int]], Path]:
    trial_rows = [
        {"left_id": "mono_alpha:enroll", "right_id": "mono_alpha:test", "label": 1},
        {"left_id": "mono_alpha:enroll", "right_id": "mono_bravo:test", "label": 0},
        {"left_id": "mono_bravo:enroll", "right_id": "mono_bravo:test", "label": 1},
        {"left_id": "mono_bravo:enroll", "right_id": "mono_alpha:test", "label": 0},
        {"left_id": "phone_charlie:enroll", "right_id": "phone_charlie:test", "label": 1},
        {"left_id": "phone_charlie:enroll", "right_id": "phone_delta:test", "label": 0},
        {"left_id": "phone_delta:enroll", "right_id": "phone_delta:test", "label": 1},
        {"left_id": "phone_delta:enroll", "right_id": "phone_charlie:test", "label": 0},
    ]
    metadata_path = tmp_path / "metadata.parquet"
    pl.DataFrame(
        [
            {
                "trial_item_id": "mono_alpha:enroll",
                "speaker_id": "mono_alpha",
                "channel": "mono",
                "role": "enrollment",
                "duration_seconds": 1.2,
            },
            {
                "trial_item_id": "mono_alpha:test",
                "speaker_id": "mono_alpha",
                "channel": "mono",
                "role": "test",
                "duration_seconds": 1.1,
            },
            {
                "trial_item_id": "mono_bravo:enroll",
                "speaker_id": "mono_bravo",
                "channel": "mono",
                "role": "enrollment",
                "duration_seconds": 1.3,
            },
            {
                "trial_item_id": "mono_bravo:test",
                "speaker_id": "mono_bravo",
                "channel": "mono",
                "role": "test",
                "duration_seconds": 1.2,
            },
            {
                "trial_item_id": "phone_charlie:enroll",
                "speaker_id": "phone_charlie",
                "channel": "phone",
                "role": "enrollment",
                "duration_seconds": 3.0,
            },
            {
                "trial_item_id": "phone_charlie:test",
                "speaker_id": "phone_charlie",
                "channel": "phone",
                "role": "test",
                "duration_seconds": 3.1,
            },
            {
                "trial_item_id": "phone_delta:enroll",
                "speaker_id": "phone_delta",
                "channel": "phone",
                "role": "enrollment",
                "duration_seconds": 2.9,
            },
            {
                "trial_item_id": "phone_delta:test",
                "speaker_id": "phone_delta",
                "channel": "phone",
                "role": "test",
                "duration_seconds": 3.2,
            },
        ]
    ).write_parquet(metadata_path)
    return trial_rows, metadata_path
