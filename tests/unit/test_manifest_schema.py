from __future__ import annotations

import json
from pathlib import Path

from kryptonite.data.schema import (
    MANIFEST_RECORD_TYPE,
    MANIFEST_SCHEMA_VERSION,
    ManifestRow,
    validate_manifest_entry,
)
from kryptonite.data.validation import build_manifest_validation_report


def test_manifest_row_serializes_canonical_fields_and_extra_metadata() -> None:
    row = ManifestRow(
        dataset="ffsvc2022-surrogate",
        source_dataset="ffsvc2022",
        speaker_id="0449",
        audio_path="datasets/ffsvc2022-surrogate/raw/dev/example.wav",
        utterance_id="ffsvc22_dev_000001",
        session_id="0449:1",
        split="train",
        channel="mono",
        sample_rate_hz=16_000,
        num_channels=1,
    )

    payload = row.to_dict(extra_fields={"original_name": "S0449_449I1M_1_0211_normal"})

    assert payload["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert payload["record_type"] == MANIFEST_RECORD_TYPE
    assert payload["source_dataset"] == "ffsvc2022"
    assert payload["session_id"] == "0449:1"
    assert payload["num_channels"] == 1
    assert payload["original_name"] == "S0449_449I1M_1_0211_normal"


def test_manifest_row_from_mapping_accepts_legacy_aliases() -> None:
    row = ManifestRow.from_mapping(
        {
            "dataset": "demo-speaker-recognition",
            "source_dataset": "demo-speaker-recognition",
            "speaker_id": "speaker_alpha",
            "audio_path": "datasets/demo-speaker-recognition/speaker_alpha/sample.wav",
            "session_index": "demo",
            "channels": 1,
        },
        require_schema_version=False,
    )

    assert row.session_id == "speaker_alpha:demo"
    assert row.num_channels == 1
    assert row.channel == "mono"


def test_validate_manifest_entry_rejects_missing_schema_and_bad_values() -> None:
    issues = validate_manifest_entry(
        {
            "dataset": "ffsvc2022-surrogate",
            "speaker_id": "0449",
            "audio_path": "datasets/ffsvc2022-surrogate/raw/dev/example.wav",
            "sample_rate_hz": 0,
        },
        require_schema_version=True,
    )
    issue_by_field = {issue.field: issue.code for issue in issues}

    assert issue_by_field["schema_version"] == "missing_field"
    assert issue_by_field["source_dataset"] == "missing_field"
    assert issue_by_field["sample_rate_hz"] == "invalid_value"


def test_manifest_validation_report_skips_trials_and_reports_invalid_rows(tmp_path: Path) -> None:
    manifests_root = tmp_path / "artifacts" / "manifests"
    manifests_root.mkdir(parents=True)
    (manifests_root / "official_dev_trials.jsonl").write_text(
        json.dumps({"label": 1, "left_audio": "a.wav", "right_audio": "b.wav"}) + "\n"
    )
    (manifests_root / "demo_manifest.jsonl").write_text(
        json.dumps(
            {
                "dataset": "demo-speaker-recognition",
                "speaker_id": "speaker_alpha",
                "audio_path": "datasets/demo-speaker-recognition/speaker_alpha/sample.wav",
            }
        )
        + "\n"
    )

    report = build_manifest_validation_report(
        project_root=tmp_path,
        manifests_root="artifacts/manifests",
    )

    assert report.passed is False
    assert report.validated_manifest_count == 1
    assert report.skipped_manifest_count == 1
    assert report.invalid_row_count == 1
    assert report.issues[0].manifest_path == "artifacts/manifests/demo_manifest.jsonl"
