from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import soundfile as sf

from kryptonite.eval import build_embedding_atlas, write_embedding_atlas_report


def test_build_embedding_atlas_projects_points_and_neighbors(tmp_path: Path) -> None:
    embeddings_path, metadata_path = _write_fixture_bundle(tmp_path)

    report = build_embedding_atlas(
        project_root=tmp_path,
        output_root="artifacts/eval/embedding-atlas",
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        title="Test Atlas",
        projection_method="cosine_pca",
        point_id_field="utterance_id",
        label_field="speaker_id",
        color_by_field="speaker_id",
        search_fields=("speaker_id", "split", "dataset"),
        audio_path_field="audio_path",
        image_path_field=None,
        neighbors=2,
    )

    assert report.summary.point_count == 6
    assert report.summary.embedding_dim == 4
    assert report.summary.projection_method == "cosine_pca"
    assert report.summary.neighbor_count == 2
    assert report.summary.distinct_label_count == 3
    point_zero = report.points[0]
    assert point_zero.audio_href is not None
    assert point_zero.label == "speaker_a"
    assert len(point_zero.neighbors) == 2
    assert point_zero.neighbors[0].label == "speaker_a"
    assert point_zero.search_text == "speaker_a | train | demo-set"


def test_write_embedding_atlas_report_emits_points_html_and_json(tmp_path: Path) -> None:
    embeddings_path, metadata_path = _write_fixture_bundle(tmp_path)
    report = build_embedding_atlas(
        project_root=tmp_path,
        output_root="artifacts/eval/embedding-atlas",
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        title="Voice Atlas",
        projection_method="pca",
        point_id_field="utterance_id",
        label_field="speaker_id",
        color_by_field="speaker_id",
        search_fields=("speaker_id", "split"),
        audio_path_field="audio_path",
        image_path_field=None,
        neighbors=3,
    )

    written = write_embedding_atlas_report(report)
    points_rows = _read_jsonl(Path(written.points_path))
    payload = json.loads(Path(written.json_path).read_text())
    html = Path(written.html_path).read_text()
    markdown = Path(written.markdown_path).read_text()

    assert Path(written.points_path).is_file()
    assert Path(written.html_path).is_file()
    assert len(points_rows) == 6
    assert payload["summary"]["point_count"] == 6
    assert "Voice Atlas" in html
    assert "Interactive embedding atlas" in html
    assert "open `embedding_atlas.html`" in markdown.lower()


def test_build_embedding_atlas_accepts_parquet_metadata(tmp_path: Path) -> None:
    embeddings_path, metadata_path = _write_fixture_bundle(tmp_path)
    parquet_path = metadata_path.with_suffix(".parquet")
    rows = _read_jsonl(metadata_path)
    pl.DataFrame(rows).write_parquet(parquet_path)

    report = build_embedding_atlas(
        project_root=tmp_path,
        output_root="artifacts/eval/embedding-atlas",
        embeddings_path=embeddings_path,
        metadata_path=parquet_path,
        title="Parquet Atlas",
        projection_method="cosine_pca",
        point_id_field="utterance_id",
        label_field="speaker_id",
        color_by_field="speaker_id",
        search_fields=("speaker_id", "split", "dataset"),
        audio_path_field="audio_path",
        image_path_field=None,
        neighbors=2,
    )

    assert report.summary.point_count == 6
    assert report.points[0].label == "speaker_a"


def _write_fixture_bundle(tmp_path: Path) -> tuple[Path, Path]:
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.98, 0.02, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.02, 0.98, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.02, 0.98, 0.0],
        ],
        dtype=np.float64,
    )
    embeddings_path = tmp_path / "artifacts" / "eval" / "demo_embeddings.npy"
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)

    audio_root = tmp_path / "datasets" / "demo"
    audio_root.mkdir(parents=True, exist_ok=True)
    metadata_path = tmp_path / "artifacts" / "manifests" / "demo_manifest.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    speakers = ["speaker_a", "speaker_a", "speaker_b", "speaker_b", "speaker_c", "speaker_c"]
    splits = ["train", "train", "train", "dev", "dev", "dev"]
    for index, (speaker, split) in enumerate(zip(speakers, splits, strict=True)):
        audio_name = f"utt_{index:03d}.wav"
        _write_tone(audio_root / audio_name, frequency_hz=220.0 + index * 35.0)
        rows.append(
            {
                "schema_version": "kryptonite.manifest.v1",
                "record_type": "utterance",
                "dataset": "demo-set",
                "source_dataset": "demo-set",
                "speaker_id": speaker,
                "utterance_id": f"utt-{index:03d}",
                "audio_path": f"datasets/demo/{audio_name}",
                "split": split,
            }
        )
    metadata_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return embeddings_path, metadata_path


def _write_tone(path: Path, *, frequency_hz: float, sample_rate_hz: int = 16_000) -> None:
    timeline = np.arange(int(sample_rate_hz * 0.4), dtype=np.float64) / float(sample_rate_hz)
    waveform = 0.3 * np.sin(2.0 * np.pi * frequency_hz * timeline)
    sf.write(path, waveform.astype(np.float32), sample_rate_hz, format="WAV")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
