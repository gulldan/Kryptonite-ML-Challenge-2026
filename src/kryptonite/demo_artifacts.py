"""Generate a tiny reproducible demo artifact set for deploy smoke checks."""

from __future__ import annotations

import json
import math
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper

from kryptonite.config import ProjectConfig
from kryptonite.data import AudioLoadRequest
from kryptonite.deployment import resolve_project_path
from kryptonite.features import FbankExtractionRequest, UtteranceChunkingRequest
from kryptonite.serve.export_boundary import (
    ExportBoundaryContract,
    build_export_boundary_contract,
    build_model_bundle_metadata,
)

from .data.manifest_artifacts import (
    build_file_artifact,
    write_manifest_inventory,
    write_tabular_artifact,
)
from .data.schema import ManifestRow
from .serve.enrollment_cache import build_enrollment_embedding_cache

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_DURATION_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class DemoClipSpec:
    speaker_id: str
    clip_name: str
    frequency_hz: float
    role: str


@dataclass(slots=True)
class GeneratedDemoArtifacts:
    dataset_root: str
    manifests_root: str
    demo_subset_root: str
    model_bundle_root: str
    enrollment_cache_root: str
    manifest_file: str
    manifest_inventory_file: str
    subset_file: str
    model_file: str
    metadata_file: str
    enrollment_embeddings_file: str
    enrollment_summary_file: str
    clip_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_root": self.dataset_root,
            "manifests_root": self.manifests_root,
            "demo_subset_root": self.demo_subset_root,
            "model_bundle_root": self.model_bundle_root,
            "enrollment_cache_root": self.enrollment_cache_root,
            "manifest_file": self.manifest_file,
            "manifest_inventory_file": self.manifest_inventory_file,
            "subset_file": self.subset_file,
            "model_file": self.model_file,
            "metadata_file": self.metadata_file,
            "enrollment_embeddings_file": self.enrollment_embeddings_file,
            "enrollment_summary_file": self.enrollment_summary_file,
            "clip_count": self.clip_count,
        }


DEMO_CLIPS: tuple[DemoClipSpec, ...] = (
    DemoClipSpec("speaker_alpha", "enroll_01.wav", 220.0, "enrollment"),
    DemoClipSpec("speaker_alpha", "enroll_02.wav", 235.0, "enrollment"),
    DemoClipSpec("speaker_alpha", "test_01.wav", 250.0, "test"),
    DemoClipSpec("speaker_bravo", "enroll_01.wav", 330.0, "enrollment"),
    DemoClipSpec("speaker_bravo", "enroll_02.wav", 345.0, "enrollment"),
    DemoClipSpec("speaker_bravo", "test_01.wav", 360.0, "test"),
)


def generate_demo_artifacts(
    *,
    config: ProjectConfig,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
) -> GeneratedDemoArtifacts:
    project_root = config.paths.project_root
    dataset_root = resolve_project_path(project_root, config.paths.dataset_root)
    manifests_root = resolve_project_path(project_root, config.paths.manifests_root)
    demo_subset_root = resolve_project_path(project_root, config.deployment.demo_subset_root)
    model_bundle_root = resolve_project_path(project_root, config.deployment.model_bundle_root)
    enrollment_cache_root = resolve_project_path(
        project_root,
        config.deployment.enrollment_cache_root,
    )

    dataset_demo_root = dataset_root / "demo-speaker-recognition"
    manifest_file = manifests_root / "demo_manifest.jsonl"
    manifest_inventory = manifests_root / "demo_manifest_inventory.json"
    subset_file = demo_subset_root / "demo_subset.json"
    model_file = model_bundle_root / "model.onnx"
    metadata_file = model_bundle_root / "metadata.json"

    for path in (
        dataset_demo_root,
        manifests_root,
        demo_subset_root / "enrollment",
        demo_subset_root / "test",
        model_bundle_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, object]] = []
    subset_entries: dict[str, list[dict[str, object]]] = {"enrollment": [], "test": []}

    for clip in DEMO_CLIPS:
        dataset_path = dataset_demo_root / clip.speaker_id / clip.clip_name
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        _write_sine_wave(
            path=dataset_path,
            frequency_hz=clip.frequency_hz,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
        )

        subset_path = demo_subset_root / clip.role / f"{clip.speaker_id}-{clip.clip_name}"
        shutil.copy2(dataset_path, subset_path)

        relative_dataset_path = _relative_to_project(dataset_path, project_root)
        relative_subset_path = _relative_to_project(subset_path, project_root)
        manifest_entry = ManifestRow(
            dataset="demo-speaker-recognition",
            source_dataset="demo-speaker-recognition",
            speaker_id=clip.speaker_id,
            utterance_id=f"{clip.speaker_id}:{Path(clip.clip_name).stem}",
            session_id=f"{clip.speaker_id}:demo",
            split="demo",
            role=clip.role,
            device="synthetic-tone",
            channel="mono",
            audio_path=relative_dataset_path,
            duration_seconds=duration_seconds,
            sample_rate_hz=sample_rate,
            num_channels=1,
        ).to_dict(extra_fields={"demo_subset_path": relative_subset_path})
        manifest_entries.append(manifest_entry)
        subset_entries[clip.role].append(
            {
                "speaker_id": clip.speaker_id,
                "audio_path": relative_subset_path,
            }
        )

    manifest_artifact = write_tabular_artifact(
        name="demo_manifest",
        kind="data_manifest",
        rows=manifest_entries,
        jsonl_path=manifest_file,
        project_root=project_root,
    )
    subset_file.write_text(json.dumps(subset_entries, indent=2, sort_keys=True))

    export_boundary = build_export_boundary_contract(
        config=config,
        inferencer_backend="feature_statistics",
        embedding_stage="demo",
        embedding_mode="mean_std",
    )
    _write_demo_model(path=model_file, export_boundary=export_boundary)
    metadata_file.write_text(
        json.dumps(
            build_model_bundle_metadata(
                config=config,
                model_file=_relative_to_project(model_file, project_root),
                enrollment_cache_compatibility_id="demo-speaker-recognition-cache-v1",
                description=(
                    "Synthetic ONNX encoder-boundary stub for strict container smoke checks. "
                    "The runtime frontend stays outside the graph; this bundle validates only "
                    "the deploy contract shape, not production SV quality."
                ),
                inferencer_backend="feature_statistics",
                embedding_stage="demo",
                embedding_mode="mean_std",
                extra_metadata={
                    "sample_rate_hz": sample_rate,
                    "structural_stub": True,
                },
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_manifest_inventory(
        dataset="demo-speaker-recognition",
        inventory_path=manifest_inventory,
        project_root=project_root,
        manifest_tables=(manifest_artifact,),
        auxiliary_files=(
            build_file_artifact(
                name="demo_subset",
                kind="metadata",
                path=subset_file,
                project_root=project_root,
            ),
        ),
    )
    enrollment_cache = build_enrollment_embedding_cache(
        project_root=project_root,
        manifest_path=manifest_file,
        output_root=enrollment_cache_root,
        model_metadata_path=metadata_file,
        audio_request=AudioLoadRequest.from_config(config.normalization, vad=config.vad),
        fbank_request=FbankExtractionRequest.from_config(config.features),
        chunking_request=UtteranceChunkingRequest.from_config(config.chunking),
        stage="demo",
        embedding_mode="mean_std",
        device=config.runtime.device,
    )

    return GeneratedDemoArtifacts(
        dataset_root=str(dataset_root),
        manifests_root=str(manifests_root),
        demo_subset_root=str(demo_subset_root),
        model_bundle_root=str(model_bundle_root),
        enrollment_cache_root=str(enrollment_cache_root),
        manifest_file=str(manifest_file),
        manifest_inventory_file=str(manifest_inventory),
        subset_file=str(subset_file),
        model_file=str(model_file),
        metadata_file=str(metadata_file),
        enrollment_embeddings_file=enrollment_cache.embeddings_path,
        enrollment_summary_file=enrollment_cache.summary_path,
        clip_count=len(DEMO_CLIPS),
    )


def _write_sine_wave(
    *,
    path: Path,
    frequency_hz: float,
    sample_rate: int,
    duration_seconds: float,
) -> None:
    amplitude = 0.35
    frame_count = int(sample_rate * duration_seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            sample = amplitude * math.sin(2.0 * math.pi * frequency_hz * index / sample_rate)
            pcm_value = int(max(-1.0, min(1.0, sample)) * 32767)
            frames.extend(pcm_value.to_bytes(2, byteorder="little", signed=True))
        handle.writeframes(bytes(frames))


def _write_demo_model(*, path: Path, export_boundary: ExportBoundaryContract) -> None:
    input_name = export_boundary.input_tensor.name
    output_name = export_boundary.output_tensor.name
    num_mel_bins = export_boundary.input_tensor.axes[-1].size
    embedding_dim = export_boundary.output_tensor.axes[-1].size
    if not isinstance(num_mel_bins, int):
        raise ValueError("Demo export boundary requires a fixed mel-bin axis.")
    if not isinstance(embedding_dim, int):
        raise ValueError("Demo export boundary requires a fixed embedding dimension.")

    nodes = [
        helper.make_node(
            "ReduceMean",
            inputs=[input_name, "reduce_axes"],
            outputs=["mean_embedding"],
            keepdims=0,
        )
    ]
    if embedding_dim == num_mel_bins:
        nodes.append(helper.make_node("Identity", inputs=["mean_embedding"], outputs=[output_name]))
    elif embedding_dim == num_mel_bins * 2:
        nodes.extend(
            [
                helper.make_node("Abs", inputs=["mean_embedding"], outputs=["abs_mean_embedding"]),
                helper.make_node(
                    "Concat",
                    inputs=["mean_embedding", "abs_mean_embedding"],
                    outputs=[output_name],
                    axis=1,
                ),
            ]
        )
    else:
        raise ValueError(
            "Demo export boundary stub supports only embedding_dim equal to "
            f"{num_mel_bins} or {num_mel_bins * 2}, got {embedding_dim}."
        )

    graph = helper.make_graph(
        nodes=nodes,
        name="demo-speaker-recognition",
        inputs=[
            helper.make_tensor_value_info(
                input_name,
                TensorProto.FLOAT,
                [1, "frames", num_mel_bins],
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                output_name,
                TensorProto.FLOAT,
                [1, embedding_dim],
            )
        ],
        initializer=[
            helper.make_tensor(
                name="reduce_axes",
                data_type=TensorProto.INT64,
                dims=[1],
                vals=[1],
            )
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="kryptonite-demo-artifacts",
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    checker.check_model(model)
    onnx.save_model(model, path)


def _relative_to_project(path: Path, project_root: str) -> str:
    root = resolve_project_path(project_root, ".")
    return str(path.resolve().relative_to(root))
