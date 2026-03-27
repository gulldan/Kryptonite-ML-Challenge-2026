from __future__ import annotations

import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.demo_artifacts import generate_demo_artifacts
from kryptonite.serve.triton_repository import (
    TritonRepositoryRequest,
    build_triton_model_repository,
    build_triton_repository_source_report,
    normalize_triton_backend_mode,
)


def test_build_triton_model_repository_from_onnx_bundle(tmp_path: Path) -> None:
    config = _build_demo_config(tmp_path)

    built = build_triton_model_repository(config=config)

    config_text = Path(built.config_path).read_text(encoding="utf-8")
    request_payload = json.loads(Path(built.smoke_request_path).read_text(encoding="utf-8"))
    readme_text = Path(built.readme_path).read_text(encoding="utf-8")

    assert built.backend_mode == "onnx"
    assert built.platform == "onnxruntime_onnx"
    assert Path(built.model_path).name == "model.onnx"
    assert 'platform: "onnxruntime_onnx"' in config_text
    assert 'name: "kryptonite_encoder"' in config_text
    assert "dims: [-1, 80]" in config_text
    assert "dims: [160]" in config_text
    assert request_payload["inputs"][0]["datatype"] == "FP32"
    assert request_payload["inputs"][0]["shape"] == [1, 12, 80]
    assert request_payload["outputs"] == [{"name": "embedding"}]
    assert "frontend still lives outside Triton" in readme_text


def test_triton_repository_source_report_requires_tensorrt_engine(tmp_path: Path) -> None:
    config = _build_demo_config(tmp_path)

    report = build_triton_repository_source_report(
        config=config,
        request=TritonRepositoryRequest(backend_mode="tensorrt"),
    )

    assert report.passed is False
    assert report.missing_required == ["source_engine_file"]


def test_build_triton_model_repository_from_tensorrt_engine(tmp_path: Path) -> None:
    config = _build_demo_config(tmp_path)
    engine_path = tmp_path / "artifacts" / "model-bundle" / "model.plan"
    engine_path.write_bytes(b"fake-plan")

    built = build_triton_model_repository(
        config=config,
        request=TritonRepositoryRequest(backend_mode="tensorrt"),
    )

    config_text = Path(built.config_path).read_text(encoding="utf-8")

    assert built.backend_mode == "tensorrt"
    assert built.platform == "tensorrt_plan"
    assert built.instance_kind == "KIND_GPU"
    assert Path(built.model_path).name == "model.plan"
    assert Path(built.model_path).read_bytes() == b"fake-plan"
    assert 'platform: "tensorrt_plan"' in config_text
    assert "kind: KIND_GPU" in config_text


def test_normalize_triton_backend_mode_rejects_unknown_value() -> None:
    try:
        normalize_triton_backend_mode("python")
    except ValueError as exc:
        assert "backend_mode must be one of" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("normalize_triton_backend_mode should reject unsupported values")


def _build_demo_config(tmp_path: Path):
    config = load_project_config(
        config_path=Path("configs/deployment/infer.toml"),
        overrides=[
            f'paths.project_root="{tmp_path}"',
            'paths.dataset_root="datasets"',
            'paths.manifests_root="artifacts/manifests"',
            'deployment.model_bundle_root="artifacts/model-bundle"',
            'deployment.demo_subset_root="artifacts/demo-subset"',
            'deployment.enrollment_cache_root="artifacts/enrollment-cache"',
        ],
    )
    generate_demo_artifacts(config=config)
    return config
