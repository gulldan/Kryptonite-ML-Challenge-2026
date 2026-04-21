from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kryptonite.runtime.enrollment_store import RuntimeEnrollmentStore


def test_runtime_enrollment_store_roundtrips_records(tmp_path: Path) -> None:
    metadata_path, metadata_payload = _write_model_metadata(
        tmp_path,
        compatibility_id="demo-speaker-recognition-cache-v1",
    )
    store = RuntimeEnrollmentStore(
        store_root=tmp_path / "artifacts" / "enrollment-cache",
        model_metadata_path=metadata_path,
        model_metadata_location="artifacts/model-bundle/metadata.json",
        model_metadata=metadata_payload,
    )

    store.upsert(
        enrollment_id="speaker-charlie",
        sample_count=2,
        embedding_dim=2,
        embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        metadata={"source": "unit-test", "tags": ["demo"]},
    )

    reopened = RuntimeEnrollmentStore(
        store_root=tmp_path / "artifacts" / "enrollment-cache",
        model_metadata_path=metadata_path,
        model_metadata_location="artifacts/model-bundle/metadata.json",
        model_metadata=metadata_payload,
    )
    records = reopened.load_records()
    summary = reopened.summary()

    assert summary.enrollment_count == 1
    assert summary.compatibility_id == "demo-speaker-recognition-cache-v1"
    assert records["speaker-charlie"].sample_count == 2
    assert records["speaker-charlie"].metadata == {"source": "unit-test", "tags": ["demo"]}
    np.testing.assert_allclose(
        records["speaker-charlie"].embedding,
        np.asarray([1.0, 0.0], dtype=np.float32),
    )


def test_runtime_enrollment_store_rejects_incompatible_model_metadata(tmp_path: Path) -> None:
    metadata_path, first_payload = _write_model_metadata(
        tmp_path,
        compatibility_id="cache-v1",
    )
    RuntimeEnrollmentStore(
        store_root=tmp_path / "artifacts" / "enrollment-cache",
        model_metadata_path=metadata_path,
        model_metadata_location="artifacts/model-bundle/metadata.json",
        model_metadata=first_payload,
    )

    _, second_payload = _write_model_metadata(
        tmp_path,
        compatibility_id="cache-v2",
    )
    with pytest.raises(ValueError, match="Runtime enrollment store compatibility mismatch"):
        RuntimeEnrollmentStore(
            store_root=tmp_path / "artifacts" / "enrollment-cache",
            model_metadata_path=metadata_path,
            model_metadata_location="artifacts/model-bundle/metadata.json",
            model_metadata=second_payload,
        )


def _write_model_metadata(
    tmp_path: Path,
    *,
    compatibility_id: str,
) -> tuple[Path, dict[str, str | int]]:
    metadata_path = tmp_path / "artifacts" / "model-bundle" / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_file": "artifacts/model-bundle/model.onnx",
        "input_name": "audio",
        "output_name": "embedding",
        "sample_rate_hz": 16000,
        "enrollment_cache_compatibility_id": compatibility_id,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path, payload
