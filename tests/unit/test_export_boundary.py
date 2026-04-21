from __future__ import annotations

from pathlib import Path

import pytest

from kryptonite.config import load_project_config
from kryptonite.runtime.export_boundary import (
    build_export_boundary_contract,
    build_model_bundle_metadata,
    load_export_boundary_from_model_metadata,
    render_export_boundary_markdown,
    validate_runtime_frontend_against_boundary,
)


def test_build_export_boundary_contract_uses_encoder_only_boundary() -> None:
    config = load_project_config(config_path=Path("configs/base.toml"))

    contract = build_export_boundary_contract(
        config=config,
        inferencer_backend="feature_statistics",
        embedding_stage="demo",
        embedding_mode="mean_std",
    )

    assert contract.boundary == "encoder_only"
    assert contract.input_tensor.name == "encoder_input"
    assert contract.input_tensor.layout == "BTF"
    assert contract.input_tensor.axes[-1].size == 80
    assert contract.output_tensor.name == "embedding"
    assert contract.output_tensor.axes[-1].size == 160
    assert contract.runtime_frontend["audio_load_request"]["target_sample_rate_hz"] == 16000
    assert "extract_fbank_before_engine" in contract.runtime_pre_engine_steps

    markdown = render_export_boundary_markdown(contract)
    assert "encoder_input" in markdown
    assert "pool_chunk_embeddings" in markdown


def test_model_bundle_metadata_round_trips_export_boundary() -> None:
    config = load_project_config(config_path=Path("configs/base.toml"))

    metadata = build_model_bundle_metadata(
        config=config,
        model_file="artifacts/model-bundle/model.onnx",
        enrollment_cache_compatibility_id="cache-v1",
        description="demo",
        inferencer_backend="feature_statistics",
        embedding_stage="demo",
        embedding_mode="mean_std",
    )

    contract = load_export_boundary_from_model_metadata(metadata)

    assert metadata["model_version"] == "baseline"
    assert metadata["input_name"] == "encoder_input"
    assert metadata["output_name"] == "embedding"
    assert metadata["inference_package"]["validated_backends"]["torch"] is True
    assert metadata["inference_package"]["validated_backends"]["onnxruntime"] is False
    assert contract.output_tensor.axes[-1].size == 160
    validate_runtime_frontend_against_boundary(config=config, contract=contract)


def test_validate_runtime_frontend_against_boundary_rejects_feature_mismatch() -> None:
    config = load_project_config(config_path=Path("configs/base.toml"))
    metadata = build_model_bundle_metadata(
        config=config,
        model_file="artifacts/model-bundle/model.onnx",
        enrollment_cache_compatibility_id="cache-v1",
        description="demo",
        inferencer_backend="feature_statistics",
        embedding_stage="demo",
        embedding_mode="mean_std",
    )
    metadata["export_boundary"]["runtime_frontend"]["features"]["num_mel_bins"] = 64

    contract = load_export_boundary_from_model_metadata(metadata)

    with pytest.raises(ValueError, match="num_mel_bins"):
        validate_runtime_frontend_against_boundary(config=config, contract=contract)
