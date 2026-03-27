from __future__ import annotations

import json
from pathlib import Path

from kryptonite.eval import (
    build_final_benchmark_pack,
    load_final_benchmark_pack_config,
    render_final_benchmark_pack_markdown,
    write_final_benchmark_pack,
)


def test_final_benchmark_pack_stages_sources_and_builds_pairwise_summary(tmp_path: Path) -> None:
    alpha_paths = _write_candidate_fixture(
        tmp_path,
        candidate_id="alpha",
        eer=0.145,
        min_dcf=0.284,
        mean_ms_per_audio=12.5,
        peak_rss_mib=512.0,
        peak_cuda_allocated_mib=2048.0,
    )
    bravo_paths = _write_candidate_fixture(
        tmp_path,
        candidate_id="bravo",
        eer=0.121,
        min_dcf=0.241,
        mean_ms_per_audio=15.0,
        peak_rss_mib=544.0,
        peak_cuda_allocated_mib=1984.0,
    )
    config_path = _write_pack_config(tmp_path, alpha_paths=alpha_paths, bravo_paths=bravo_paths)

    config = load_final_benchmark_pack_config(config_path=config_path)
    report = build_final_benchmark_pack(config, config_path=config_path)
    written = write_final_benchmark_pack(report)

    assert report.summary.candidate_count == 2
    assert report.summary.pairwise_comparison_count == 1
    assert report.summary.best_eer_candidate_id == "bravo"
    assert report.summary.best_min_dcf_candidate_id == "bravo"
    assert report.summary.lowest_latency_candidate_id == "alpha"
    assert report.summary.lowest_process_rss_candidate_id == "alpha"
    assert report.summary.lowest_cuda_allocated_candidate_id == "bravo"

    comparison = report.pairwise_comparisons[0]
    assert comparison.left_candidate_id == "alpha"
    assert comparison.right_candidate_id == "bravo"
    assert comparison.better_quality_candidate_id == "bravo"
    assert comparison.lower_latency_candidate_id == "alpha"
    assert comparison.lower_process_rss_candidate_id == "alpha"
    assert comparison.lower_cuda_allocated_candidate_id == "bravo"
    assert comparison.eer_delta_right_minus_left < 0.0
    assert comparison.latency_delta_ms_per_audio_right_minus_left is not None
    assert comparison.latency_delta_ms_per_audio_right_minus_left > 0.0

    source_paths = {
        artifact.copied_path
        for candidate in report.candidates
        for artifact in candidate.source_artifacts
    }
    assert any(path.endswith("verification_eval_report.json") for path in source_paths)
    assert any(path.endswith("model_bundle_metadata.json") for path in source_paths)
    assert any(path.endswith("config_01_train.toml") for path in source_paths)
    assert any(path.endswith("supporting_01_selection.json") for path in source_paths)

    assert Path(written.report_json_path).is_file()
    assert Path(written.report_markdown_path).is_file()
    assert Path(written.candidate_jsonl_path).is_file()
    assert Path(written.pairwise_jsonl_path).is_file()

    markdown = render_final_benchmark_pack_markdown(report)
    assert "Pairwise Comparisons" in markdown
    assert "Peak RSS MiB" in markdown

    written_payload = json.loads(Path(written.report_json_path).read_text())
    assert written_payload["summary"]["best_eer_candidate_id"] == "bravo"
    assert len(written_payload["candidates"]) == 2


def test_final_benchmark_pack_config_rejects_duplicate_candidate_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "final-benchmark-pack.toml"
    config_path.write_text(
        "\n".join(
            [
                'title = "dup"',
                'summary = "dup"',
                f'output_root = "{(tmp_path / "out").as_posix()}"',
                "",
                "[[candidate]]",
                'candidate_id = "same"',
                'label = "A"',
                'family = "campp"',
                f'verification_report_path = "{(tmp_path / "a.json").as_posix()}"',
                f'stress_report_path = "{(tmp_path / "b.json").as_posix()}"',
                f'model_bundle_metadata_path = "{(tmp_path / "c.json").as_posix()}"',
                f'config_paths = ["{(tmp_path / "x.toml").as_posix()}"]',
                "",
                "[[candidate]]",
                'candidate_id = "same"',
                'label = "B"',
                'family = "eres2netv2"',
                f'verification_report_path = "{(tmp_path / "d.json").as_posix()}"',
                f'stress_report_path = "{(tmp_path / "e.json").as_posix()}"',
                f'model_bundle_metadata_path = "{(tmp_path / "f.json").as_posix()}"',
                f'config_paths = ["{(tmp_path / "y.toml").as_posix()}"]',
                "",
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_final_benchmark_pack_config(config_path=config_path)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("Expected duplicate candidate ids to raise ValueError.")


def _write_candidate_fixture(
    tmp_path: Path,
    *,
    candidate_id: str,
    eer: float,
    min_dcf: float,
    mean_ms_per_audio: float,
    peak_rss_mib: float,
    peak_cuda_allocated_mib: float,
) -> dict[str, Path]:
    candidate_root = tmp_path / candidate_id
    candidate_root.mkdir(parents=True, exist_ok=True)

    verification_report_path = candidate_root / "verification_eval_report.json"
    verification_report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "metrics": {
                        "trial_count": 128,
                        "positive_count": 64,
                        "negative_count": 64,
                        "eer": eer,
                        "eer_threshold": 0.51,
                        "min_dcf": min_dcf,
                        "min_dcf_threshold": 0.61,
                    },
                    "score_statistics": {
                        "mean_positive_score": 0.81,
                        "mean_negative_score": 0.22,
                        "score_gap": 0.59,
                    },
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    threshold_calibration_path = candidate_root / "verification_threshold_calibration.json"
    threshold_calibration_path.write_text(
        json.dumps(
            {
                "global_profiles": [
                    {"name": "balanced", "threshold": 0.51},
                    {"name": "demo", "threshold": 0.60},
                    {"name": "production", "threshold": 0.68},
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    stress_report_path = candidate_root / "inference_stress_report.json"
    stress_report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "passed": True,
                    "control_ordering_passed": True,
                    "long_audio_chunking_observed": True,
                },
                "memory": {
                    "peak_process_rss_mib": peak_rss_mib,
                    "peak_process_rss_delta_mib": 24.0,
                    "peak_cuda_allocated_mib": peak_cuda_allocated_mib,
                    "peak_cuda_reserved_mib": peak_cuda_allocated_mib + 128.0,
                },
                "batch_bursts": [
                    {
                        "batch_size": 1,
                        "status": "passed",
                        "mean_ms_per_audio": mean_ms_per_audio + 2.0,
                    },
                    {
                        "batch_size": 8,
                        "status": "passed",
                        "mean_ms_per_audio": mean_ms_per_audio,
                    },
                ],
                "hard_limits": {
                    "validated_stage": "eval",
                    "largest_validated_batch_size": 8,
                    "largest_validated_total_chunk_count": 40,
                    "max_validated_duration_seconds": 12.0,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    metadata_path = candidate_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_version": f"{candidate_id}-v1",
                "model_file": f"artifacts/{candidate_id}/model.onnx",
                "input_name": "encoder_input",
                "output_name": "embedding",
                "sample_rate_hz": 16000,
                "enrollment_cache_compatibility_id": f"{candidate_id}-cache-v1",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    export_boundary_path = candidate_root / "export_boundary.json"
    export_boundary_path.write_text(
        json.dumps(
            {
                "boundary": "encoder_only",
                "export_profile": "baseline",
                "frontend_location": "runtime",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    train_config_path = candidate_root / "train.toml"
    train_config_path.write_text("learning_rate = 0.001\n", encoding="utf-8")
    infer_config_path = candidate_root / "infer.toml"
    infer_config_path.write_text('device = "cuda"\n', encoding="utf-8")
    supporting_path = candidate_root / "selection.json"
    supporting_path.write_text('{"winner": true}\n', encoding="utf-8")

    return {
        "verification_report": verification_report_path,
        "threshold_calibration": threshold_calibration_path,
        "stress_report": stress_report_path,
        "metadata": metadata_path,
        "export_boundary": export_boundary_path,
        "train_config": train_config_path,
        "infer_config": infer_config_path,
        "supporting": supporting_path,
    }


def _write_pack_config(
    tmp_path: Path,
    *,
    alpha_paths: dict[str, Path],
    bravo_paths: dict[str, Path],
) -> Path:
    output_root = tmp_path / "artifacts" / "benchmark-pack"
    config_path = tmp_path / "final-benchmark-pack.toml"
    alpha_config_paths = (
        f'config_paths = ["{alpha_paths["train_config"].as_posix()}", '
        f'"{alpha_paths["infer_config"].as_posix()}"]'
    )
    bravo_config_paths = (
        f'config_paths = ["{bravo_paths["train_config"].as_posix()}", '
        f'"{bravo_paths["infer_config"].as_posix()}"]'
    )
    config_path.write_text(
        "\n".join(
            [
                'title = "Release benchmark pack"',
                'summary = "Frozen comparison for final candidates."',
                f'output_root = "{output_root.as_posix()}"',
                'notes = ["carry exact configs"]',
                "",
                "[[candidate]]",
                'candidate_id = "alpha"',
                'label = "Alpha"',
                'family = "campp"',
                f'verification_report_path = "{alpha_paths["verification_report"].as_posix()}"',
                f'threshold_calibration_path = "{alpha_paths["threshold_calibration"].as_posix()}"',
                f'stress_report_path = "{alpha_paths["stress_report"].as_posix()}"',
                f'model_bundle_metadata_path = "{alpha_paths["metadata"].as_posix()}"',
                f'export_boundary_path = "{alpha_paths["export_boundary"].as_posix()}"',
                alpha_config_paths,
                f'supporting_paths = ["{alpha_paths["supporting"].as_posix()}"]',
                'notes = ["current fastest candidate"]',
                "",
                "[[candidate]]",
                'candidate_id = "bravo"',
                'label = "Bravo"',
                'family = "eres2netv2"',
                f'verification_report_path = "{bravo_paths["verification_report"].as_posix()}"',
                f'threshold_calibration_path = "{bravo_paths["threshold_calibration"].as_posix()}"',
                f'stress_report_path = "{bravo_paths["stress_report"].as_posix()}"',
                f'model_bundle_metadata_path = "{bravo_paths["metadata"].as_posix()}"',
                f'export_boundary_path = "{bravo_paths["export_boundary"].as_posix()}"',
                bravo_config_paths,
                f'supporting_paths = ["{bravo_paths["supporting"].as_posix()}"]',
                'notes = ["current strongest quality candidate"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path
