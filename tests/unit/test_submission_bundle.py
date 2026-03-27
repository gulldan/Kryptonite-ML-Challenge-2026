from __future__ import annotations

import json
from pathlib import Path

from kryptonite.serve import (
    build_submission_bundle,
    build_submission_bundle_source_report,
    load_submission_bundle_config,
    render_submission_bundle_readme,
    write_submission_bundle,
)


def test_submission_bundle_stages_release_handoff_and_archive(tmp_path: Path) -> None:
    paths = _write_candidate_fixture(tmp_path)
    config_path = _write_config(tmp_path, paths=paths, bundle_mode="candidate")

    config = load_submission_bundle_config(config_path=config_path)
    report = build_submission_bundle(config, config_path=config_path)
    written = write_submission_bundle(report, create_archive=config.create_archive)

    assert report.summary.bundle_mode == "candidate"
    assert report.summary.model_version == "campp-prod-v3"
    assert report.summary.threshold_calibration_included is True
    assert report.summary.tensorrt_plan_included is True
    assert report.summary.triton_repository_included is True
    assert report.summary.checkpoint_count == 1
    assert report.summary.benchmark_artifact_count == 2
    assert report.summary.structural_stub is False

    staged_paths = {artifact.staged_path for artifact in report.artifacts}
    assert "model/model.onnx" in staged_paths
    assert "model/model.plan" in staged_paths
    assert "thresholds/verification_threshold_calibration.json" in staged_paths
    assert "checkpoints/checkpoint_01_campp.ckpt" in staged_paths
    assert "triton-model-repository" in staged_paths

    readme_text = Path(written.readme_path).read_text(encoding="utf-8")
    assert "campp-prod-v3" in readme_text
    assert "scripts/infer_smoke.py" in readme_text
    assert "scripts/triton_infer_smoke.py" in readme_text

    payload = json.loads(Path(written.report_json_path).read_text(encoding="utf-8"))
    assert payload["summary"]["checkpoint_count"] == 1
    assert payload["summary"]["tensorrt_plan_included"] is True

    assert Path(written.report_markdown_path).is_file()
    assert written.archive_path is not None
    assert Path(written.archive_path).is_file()


def test_submission_bundle_smoke_mode_allows_missing_candidate_only_inputs(tmp_path: Path) -> None:
    paths = _write_candidate_fixture(tmp_path)
    config_path = _write_config(
        tmp_path,
        paths=paths,
        bundle_mode="smoke",
        include_benchmark=False,
        include_threshold=False,
        include_checkpoint=False,
        include_tensorrt=False,
    )

    config = load_submission_bundle_config(config_path=config_path)
    report = build_submission_bundle(config, config_path=config_path)

    source_report = build_submission_bundle_source_report(config, project_root=tmp_path)
    assert source_report.passed is True
    assert report.summary.bundle_mode == "smoke"
    assert report.summary.threshold_calibration_included is False
    assert report.summary.tensorrt_plan_included is False
    assert "Smoke mode allows candidate-only artifacts" in render_submission_bundle_readme(report)


def test_submission_bundle_candidate_config_requires_threshold_and_checkpoint(
    tmp_path: Path,
) -> None:
    paths = _write_candidate_fixture(tmp_path)
    config_path = _write_config(
        tmp_path,
        paths=paths,
        bundle_mode="candidate",
        include_benchmark=True,
        include_threshold=False,
        include_checkpoint=False,
    )

    try:
        load_submission_bundle_config(config_path=config_path)
    except ValueError as exc:
        assert "threshold_calibration_path" in str(exc)
    else:
        raise AssertionError("candidate mode should require threshold_calibration_path.")


def _write_candidate_fixture(tmp_path: Path) -> dict[str, Path]:
    docs_root = tmp_path / "docs"
    configs_root = tmp_path / "configs" / "deployment"
    artifacts_root = tmp_path / "artifacts"
    docs_root.mkdir(parents=True, exist_ok=True)
    configs_root.mkdir(parents=True, exist_ok=True)
    (artifacts_root / "model-bundle").mkdir(parents=True, exist_ok=True)
    (artifacts_root / "demo-subset" / "enrollment").mkdir(parents=True, exist_ok=True)
    (artifacts_root / "triton-model-repository" / "kryptonite_encoder").mkdir(
        parents=True, exist_ok=True
    )

    repository_readme = tmp_path / "README.md"
    repository_readme.write_text("# Repo\n", encoding="utf-8")
    model_card = docs_root / "model-card.md"
    model_card.write_text("# Model Card\n", encoding="utf-8")
    runbook = docs_root / "release-runbook.md"
    runbook.write_text("# Release Runbook\n", encoding="utf-8")
    triton_doc = docs_root / "triton-deployment.md"
    triton_doc.write_text("# Triton\n", encoding="utf-8")
    benchmark_md = artifacts_root / "benchmark-pack" / "final" / "final_benchmark_pack.md"
    benchmark_md.parent.mkdir(parents=True, exist_ok=True)
    benchmark_md.write_text("# Benchmark Pack\n", encoding="utf-8")
    benchmark_json = benchmark_md.with_suffix(".json")
    benchmark_json.write_text('{"winner":"campp"}\n', encoding="utf-8")
    infer_toml = configs_root / "infer.toml"
    infer_toml.write_text('device = "cpu"\n', encoding="utf-8")
    infer_gpu_toml = configs_root / "infer-gpu.toml"
    infer_gpu_toml.write_text('device = "cuda"\n', encoding="utf-8")
    metadata = artifacts_root / "model-bundle" / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "model_version": "campp-prod-v3",
                "model_file": "artifacts/model-bundle/model.onnx",
                "input_name": "encoder_input",
                "output_name": "embedding",
                "sample_rate_hz": 16000,
                "structural_stub": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    onnx_model = artifacts_root / "model-bundle" / "model.onnx"
    onnx_model.write_bytes(b"fake-onnx")
    tensorrt_plan = artifacts_root / "model-bundle" / "model.plan"
    tensorrt_plan.write_bytes(b"fake-plan")
    threshold = artifacts_root / "release" / "current" / "verification_threshold_calibration.json"
    threshold.parent.mkdir(parents=True, exist_ok=True)
    threshold.write_text(
        '{"global_profiles":[{"name":"demo","threshold":0.6}]}\n',
        encoding="utf-8",
    )
    checkpoint = artifacts_root / "final" / "campp.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    demo_manifest = artifacts_root / "demo-subset" / "demo_subset.json"
    demo_manifest.write_text('{"enrollment":[],"test":[]}\n', encoding="utf-8")
    demo_audio = artifacts_root / "demo-subset" / "enrollment" / "speaker_alpha.wav"
    demo_audio.write_bytes(b"wav")
    triton_readme = artifacts_root / "triton-model-repository" / "README.md"
    triton_readme.write_text("# Triton Repo\n", encoding="utf-8")
    triton_config = (
        artifacts_root / "triton-model-repository" / "kryptonite_encoder" / "config.pbtxt"
    )
    triton_config.write_text('name: "kryptonite_encoder"\n', encoding="utf-8")
    supporting = artifacts_root / "export-boundary.json"
    supporting.write_text('{"boundary":"encoder_only"}\n', encoding="utf-8")

    return {
        "repository_readme": repository_readme,
        "model_card": model_card,
        "runbook": runbook,
        "triton_doc": triton_doc,
        "benchmark_md": benchmark_md,
        "benchmark_json": benchmark_json,
        "infer_toml": infer_toml,
        "infer_gpu_toml": infer_gpu_toml,
        "metadata": metadata,
        "onnx_model": onnx_model,
        "tensorrt_plan": tensorrt_plan,
        "threshold": threshold,
        "checkpoint": checkpoint,
        "demo_root": artifacts_root / "demo-subset",
        "triton_repo": artifacts_root / "triton-model-repository",
        "supporting": supporting,
    }


def _write_config(
    tmp_path: Path,
    *,
    paths: dict[str, Path],
    bundle_mode: str,
    include_benchmark: bool = True,
    include_threshold: bool = True,
    include_checkpoint: bool = True,
    include_tensorrt: bool = True,
) -> Path:
    benchmark_paths = (
        "benchmark_paths = "
        f'["{paths["benchmark_md"].as_posix()}", "{paths["benchmark_json"].as_posix()}"]'
        if include_benchmark
        else "benchmark_paths = []"
    )
    checkpoint_paths = (
        f'checkpoint_paths = ["{paths["checkpoint"].as_posix()}"]'
        if include_checkpoint
        else "checkpoint_paths = []"
    )
    tensorrt_plan_path = (
        f'tensorrt_plan_path = "{paths["tensorrt_plan"].as_posix()}"'
        if include_tensorrt
        else 'tensorrt_plan_path = ""'
    )
    threshold_path = (
        f'threshold_calibration_path = "{paths["threshold"].as_posix()}"'
        if include_threshold
        else 'threshold_calibration_path = ""'
    )

    config_path = tmp_path / "submission-bundle.toml"
    config_path.write_text(
        "\n".join(
            [
                'title = "Release bundle"',
                'bundle_id = "campp-v3"',
                f'bundle_mode = "{bundle_mode}"',
                'summary = "Frozen handoff bundle"',
                "output_root = "
                f'"{(tmp_path / "artifacts" / "release-bundles" / "campp-v3").as_posix()}"',
                "create_archive = true",
                f"require_tensorrt_plan = {'true' if include_tensorrt else 'false'}",
                f'repository_readme_path = "{paths["repository_readme"].as_posix()}"',
                f'model_card_path = "{paths["model_card"].as_posix()}"',
                f'runbook_path = "{paths["runbook"].as_posix()}"',
                f'documentation_paths = ["{paths["triton_doc"].as_posix()}"]',
                benchmark_paths,
                (
                    f'config_paths = ["{paths["infer_toml"].as_posix()}", '
                    f'"{paths["infer_gpu_toml"].as_posix()}"]'
                ),
                checkpoint_paths,
                f'supporting_paths = ["{paths["supporting"].as_posix()}"]',
                f'model_bundle_metadata_path = "{paths["metadata"].as_posix()}"',
                f'onnx_model_path = "{paths["onnx_model"].as_posix()}"',
                tensorrt_plan_path,
                threshold_path,
                f'demo_assets_root = "{paths["demo_root"].as_posix()}"',
                f'triton_repository_root = "{paths["triton_repo"].as_posix()}"',
                'notes = ["bundle smoke"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path
