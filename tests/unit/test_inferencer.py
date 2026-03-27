from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import kryptonite.serve.runtime as serve_runtime
from kryptonite.config import load_project_config
from kryptonite.demo_artifacts import generate_demo_artifacts
from kryptonite.serve import Inferencer


def test_inferencer_embeds_audio_paths_and_verifies_against_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _build_demo_config(tmp_path)
    _patch_runtime_probes(monkeypatch)

    inferencer = Inferencer.from_config(config=config)

    embed_payload = inferencer.embed_audio_paths(
        audio_paths=["artifacts/demo-subset/test/speaker_alpha-test_01.wav"]
    )
    alpha_verify = inferencer.verify_audio_paths(
        enrollment_id="speaker_alpha",
        audio_paths=["artifacts/demo-subset/test/speaker_alpha-test_01.wav"],
    )
    bravo_verify = inferencer.verify_audio_paths(
        enrollment_id="speaker_alpha",
        audio_paths=["artifacts/demo-subset/test/speaker_bravo-test_01.wav"],
    )
    benchmark_payload = inferencer.benchmark_audio_paths(
        audio_paths=[
            "artifacts/demo-subset/test/speaker_alpha-test_01.wav",
            "artifacts/demo-subset/test/speaker_bravo-test_01.wav",
        ],
        iterations=2,
        warmup_iterations=0,
    )

    assert embed_payload["mode"] == "embed"
    assert embed_payload["embedding_dim"] == 160
    assert embed_payload["item_count"] == 1
    assert len(embed_payload["items"][0]["embedding"]) == 160
    assert embed_payload["backend"]["implementation"] == "feature_statistics"

    assert alpha_verify["probe_count"] == 1
    assert alpha_verify["scores"][0] > bravo_verify["scores"][0]
    assert alpha_verify["probe_items"][0]["chunk_count"] >= 1

    assert benchmark_payload["mode"] == "benchmark"
    assert benchmark_payload["audio_count"] == 2
    assert benchmark_payload["iterations"] == 2
    assert benchmark_payload["mean_ms_per_audio"] >= 0.0


def test_inferencer_rejects_model_bundle_with_mismatched_export_boundary(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _build_demo_config(tmp_path)
    _patch_runtime_probes(monkeypatch)

    metadata_path = tmp_path / "artifacts" / "model-bundle" / "metadata.json"
    metadata_payload = json.loads(metadata_path.read_text())
    metadata_payload["export_boundary"]["runtime_frontend"]["audio_load_request"][
        "target_sample_rate_hz"
    ] = 8000
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, sort_keys=True))

    with pytest.raises(ValueError, match="Model bundle export boundary mismatch"):
        Inferencer.from_config(config=config)


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


def _patch_runtime_probes(monkeypatch) -> None:
    def fake_load_module(module_name: str) -> object:
        if module_name == "onnxruntime":
            return SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
        raise ImportError(f"{module_name} missing")

    monkeypatch.setattr(serve_runtime, "_load_module", fake_load_module)
    monkeypatch.setattr(serve_runtime, "_distribution_version", lambda _: "1.0.0")
