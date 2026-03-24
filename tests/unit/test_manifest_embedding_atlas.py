from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from kryptonite.data import AudioLoadRequest
from kryptonite.eval import build_embedding_atlas, export_manifest_fbank_embeddings
from kryptonite.features import FbankExtractionRequest, UtteranceChunkingRequest


def test_export_manifest_fbank_embeddings_writes_npz_and_metadata_sidecars(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest_fixture(tmp_path)

    exported = export_manifest_fbank_embeddings(
        project_root=tmp_path,
        manifest_path=manifest_path,
        output_root="artifacts/eval/embedding-atlas/dev-fixture",
        audio_request=AudioLoadRequest(target_sample_rate_hz=16_000, target_channels=1),
        fbank_request=FbankExtractionRequest(sample_rate_hz=16_000, num_mel_bins=16),
        chunking_request=UtteranceChunkingRequest(),
        stage="eval",
        embedding_mode="mean_std",
        device="cpu",
        max_per_speaker=1,
    )

    npz = np.load(exported.embeddings_path)
    assert npz["embeddings"].shape == (2, 32)
    assert npz["point_ids"].shape == (2,)

    metadata_rows = _read_jsonl(Path(exported.metadata_jsonl_path))
    assert len(metadata_rows) == 2
    assert metadata_rows[0]["atlas_point_id"] == "utt-000"
    assert metadata_rows[0]["embedding_mode"] == "mean_std"
    assert Path(exported.metadata_parquet_path).is_file()
    assert Path(exported.report_path).is_file()
    assert exported.summary.point_count == 2
    assert exported.summary.embedding_dim == 32


def test_exported_manifest_embeddings_feed_the_atlas(tmp_path: Path) -> None:
    manifest_path = _write_manifest_fixture(tmp_path)

    exported = export_manifest_fbank_embeddings(
        project_root=tmp_path,
        manifest_path=manifest_path,
        output_root="artifacts/eval/embedding-atlas/dev-fixture",
        audio_request=AudioLoadRequest(target_sample_rate_hz=16_000, target_channels=1),
        fbank_request=FbankExtractionRequest(sample_rate_hz=16_000, num_mel_bins=16),
        chunking_request=UtteranceChunkingRequest(),
        stage="eval",
        embedding_mode="mean",
        device="cpu",
        max_rows=3,
    )

    report = build_embedding_atlas(
        project_root=tmp_path,
        output_root="artifacts/eval/embedding-atlas/dev-fixture",
        embeddings_path=exported.embeddings_path,
        metadata_path=exported.metadata_parquet_path,
        title="Fixture Atlas",
        projection_method="cosine_pca",
        point_id_field="atlas_point_id",
        label_field="speaker_id",
        color_by_field="speaker_id",
        search_fields=("speaker_id", "capture_condition", "split"),
        audio_path_field="audio_path",
        image_path_field=None,
        neighbors=2,
        embeddings_key="embeddings",
        ids_key="point_ids",
    )

    assert report.summary.point_count == 3
    assert report.summary.embedding_dim == 16
    assert report.points[0].neighbors
    assert "capture_a" in report.points[0].search_text


def _write_manifest_fixture(tmp_path: Path) -> Path:
    audio_root = tmp_path / "datasets" / "fixture"
    audio_root.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "artifacts" / "manifests" / "fixture_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    speakers = ["speaker_a", "speaker_a", "speaker_b", "speaker_b"]
    freqs = [220.0, 233.0, 330.0, 347.0]
    for index, (speaker, freq) in enumerate(zip(speakers, freqs, strict=True)):
        audio_name = f"utt_{index:03d}.wav"
        _write_tone(audio_root / audio_name, frequency_hz=freq)
        rows.append(
            {
                "schema_version": "kryptonite.manifest.v1",
                "record_type": "utterance",
                "dataset": "fixture-set",
                "source_dataset": "fixture-set",
                "speaker_id": speaker,
                "utterance_id": f"utt-{index:03d}",
                "audio_path": f"datasets/fixture/{audio_name}",
                "split": "dev",
                "capture_condition": f"capture_{'a' if index < 2 else 'b'}",
            }
        )
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest_path


def _write_tone(path: Path, *, frequency_hz: float, sample_rate_hz: int = 16_000) -> None:
    timeline = np.arange(int(sample_rate_hz * 0.5), dtype=np.float64) / float(sample_rate_hz)
    waveform = 0.3 * np.sin(2.0 * np.pi * frequency_hz * timeline)
    sf.write(path, waveform.astype(np.float32), sample_rate_hz, format="WAV")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
