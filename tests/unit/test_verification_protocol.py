from __future__ import annotations

import json
from pathlib import Path

from kryptonite.eval import (
    build_verification_protocol_report,
    load_verification_protocol_config,
    write_verification_protocol_report,
)


def test_build_verification_protocol_report_summarizes_clean_and_prod_bundles(
    tmp_path: Path,
) -> None:
    clean_manifest = tmp_path / "artifacts" / "manifests" / "synthetic" / "dev_manifest.jsonl"
    clean_manifest.parent.mkdir(parents=True, exist_ok=True)
    clean_manifest.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in (
                {
                    "speaker_id": "spk-a",
                    "utterance_id": "spk-a:1",
                    "audio_path": "datasets/synthetic/spk-a-1.wav",
                    "duration_seconds": 1.2,
                    "dataset": "synthetic-clean",
                    "channel": "mono",
                    "silence_ratio": 0.12,
                },
                {
                    "speaker_id": "spk-a",
                    "utterance_id": "spk-a:2",
                    "audio_path": "datasets/synthetic/spk-a-2.wav",
                    "duration_seconds": 3.1,
                    "dataset": "synthetic-clean",
                    "channel": "mono",
                    "silence_ratio": 0.15,
                },
                {
                    "speaker_id": "spk-b",
                    "utterance_id": "spk-b:1",
                    "audio_path": "datasets/synthetic/spk-b-1.wav",
                    "duration_seconds": 1.4,
                    "dataset": "synthetic-clean",
                    "channel": "phone",
                    "silence_ratio": 0.48,
                },
                {
                    "speaker_id": "spk-b",
                    "utterance_id": "spk-b:2",
                    "audio_path": "datasets/synthetic/spk-b-2.wav",
                    "duration_seconds": 2.8,
                    "dataset": "synthetic-clean",
                    "channel": "phone",
                    "silence_ratio": 0.52,
                },
            )
        ),
        encoding="utf-8",
    )

    official_trials = clean_manifest.parent / "official_dev_trials.jsonl"
    official_trials.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in (
                {"label": 1, "left_audio": "spk-a-1.wav", "right_audio": "spk-a-2.wav"},
                {"label": 0, "left_audio": "spk-a-1.wav", "right_audio": "spk-b-1.wav"},
            )
        ),
        encoding="utf-8",
    )

    prod_manifest = (
        tmp_path / "artifacts" / "eval" / "corrupted-dev-suites" / "dev_codec" / "manifest.jsonl"
    )
    prod_manifest.parent.mkdir(parents=True, exist_ok=True)
    prod_manifest.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in (
                {
                    "speaker_id": "spk-a",
                    "utterance_id": "spk-a:1",
                    "audio_path": "artifacts/eval/corrupted-dev-suites/dev_codec/audio/spk-a-1.wav",
                    "duration_seconds": 1.2,
                    "dataset": "synthetic-dev_codec",
                    "channel": "mono",
                    "silence_ratio": 0.32,
                    "corruption_family": "codec",
                    "corruption_suite": "dev_codec",
                    "corruption_severity": "medium",
                    "corruption_metadata": {
                        "codec_family": "telephony",
                        "codec_name": "pcm_mulaw",
                    },
                },
                {
                    "speaker_id": "spk-b",
                    "utterance_id": "spk-b:1",
                    "audio_path": "artifacts/eval/corrupted-dev-suites/dev_codec/audio/spk-b-1.wav",
                    "duration_seconds": 1.4,
                    "dataset": "synthetic-dev_codec",
                    "channel": "phone",
                    "silence_ratio": 0.35,
                    "corruption_family": "codec",
                    "corruption_suite": "dev_codec",
                    "corruption_severity": "medium",
                    "corruption_metadata": {
                        "codec_family": "telephony",
                        "codec_name": "pcm_mulaw",
                    },
                },
            )
        ),
        encoding="utf-8",
    )
    prod_trials = prod_manifest.parent / "official_dev_trials.jsonl"
    prod_trials.write_text(
        json.dumps(
            {"label": 0, "left_audio": "spk-a-1.wav", "right_audio": "spk-b-1.wav"},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    catalog_path = (
        tmp_path
        / "artifacts"
        / "eval"
        / "corrupted-dev-suites"
        / "corrupted_dev_suites_catalog.json"
    )
    catalog_path.write_text(
        json.dumps(
            {
                "suites": [
                    {
                        "suite_id": "dev_codec",
                        "family": "codec",
                        "description": "Codec stress suite",
                        "manifest_path": (
                            "artifacts/eval/corrupted-dev-suites/dev_codec/manifest.jsonl"
                        ),
                        "trial_manifest_paths": [
                            (
                                "artifacts/eval/corrupted-dev-suites/dev_codec/"
                                "official_dev_trials.jsonl"
                            )
                        ],
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "verification-protocol.toml"
    config_path.write_text(
        "\n".join(
            (
                'title = "Synthetic Verification Protocol"',
                'ticket_id = "KVA-481"',
                'protocol_id = "synthetic-verification-protocol"',
                'summary = "Synthetic snapshot for protocol tests."',
                'output_root = "artifacts/eval/verification-protocol"',
                (
                    "required_slice_fields = ["
                    '"duration_bucket", "codec_slice", "silence_ratio_bucket"]'
                ),
                (
                    "corrupted_suite_catalog_path = "
                    '"artifacts/eval/corrupted-dev-suites/'
                    'corrupted_dev_suites_catalog.json"'
                ),
                "validation_commands = [",
                '  "uv run python scripts/build_verification_protocol.py '
                '--config verification-protocol.toml",',
                "]",
                "",
                "[[clean_sets]]",
                'bundle_id = "official-dev-reference"',
                'stage = "dev"',
                'description = "Synthetic clean dev bundle."',
                'trial_manifest_path = "artifacts/manifests/synthetic/official_dev_trials.jsonl"',
                'metadata_manifest_path = "artifacts/manifests/synthetic/dev_manifest.jsonl"',
                'notes = ["clean"]',
            )
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_verification_protocol_config(config_path=config_path)
    report = build_verification_protocol_report(
        config,
        config_path=config_path,
        project_root=tmp_path,
    )
    written = write_verification_protocol_report(
        report,
        output_root=tmp_path / "artifacts" / "eval" / "verification-protocol",
    )

    assert report.summary.clean_bundle_count == 1
    assert report.summary.production_bundle_count == 1
    assert report.summary.missing_required_slice_fields == ()
    assert report.clean_bundles[0].trial_sources[0].trial_count == 2
    assert report.production_bundles[0].available_slice_fields == (
        "duration_bucket",
        "codec_slice",
        "silence_ratio_bucket",
    )
    assert Path(written.report_json_path).is_file()
    assert Path(written.report_markdown_path).is_file()
