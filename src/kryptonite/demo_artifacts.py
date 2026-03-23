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
from kryptonite.deployment import resolve_project_path

from .data.schema import ManifestRow

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
    manifest_file: str
    subset_file: str
    model_file: str
    metadata_file: str
    clip_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_root": self.dataset_root,
            "manifests_root": self.manifests_root,
            "demo_subset_root": self.demo_subset_root,
            "model_bundle_root": self.model_bundle_root,
            "manifest_file": self.manifest_file,
            "subset_file": self.subset_file,
            "model_file": self.model_file,
            "metadata_file": self.metadata_file,
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

    dataset_demo_root = dataset_root / "demo-speaker-recognition"
    manifest_file = manifests_root / "demo_manifest.jsonl"
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

    manifest_file.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in manifest_entries)
    )
    subset_file.write_text(json.dumps(subset_entries, indent=2, sort_keys=True))

    _write_demo_model(path=model_file)
    metadata_file.write_text(
        json.dumps(
            {
                "model_file": _relative_to_project(model_file, project_root),
                "input_name": "audio",
                "output_name": "embedding",
                "sample_rate_hz": sample_rate,
                "description": (
                    "Synthetic ONNX demo bundle for strict container smoke checks. "
                    "This is a packaging/runtime validation artifact, not a production SV model."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )

    return GeneratedDemoArtifacts(
        dataset_root=str(dataset_root),
        manifests_root=str(manifests_root),
        demo_subset_root=str(demo_subset_root),
        model_bundle_root=str(model_bundle_root),
        manifest_file=str(manifest_file),
        subset_file=str(subset_file),
        model_file=str(model_file),
        metadata_file=str(metadata_file),
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


def _write_demo_model(*, path: Path) -> None:
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Identity",
                inputs=["audio"],
                outputs=["embedding"],
            )
        ],
        name="demo-speaker-recognition",
        inputs=[
            helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, DEFAULT_SAMPLE_RATE])
        ],
        outputs=[
            helper.make_tensor_value_info(
                "embedding",
                TensorProto.FLOAT,
                [1, DEFAULT_SAMPLE_RATE],
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
