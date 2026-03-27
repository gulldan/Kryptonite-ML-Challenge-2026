"""Machine-readable contract between the runtime frontend and exportable encoder graphs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, cast

from kryptonite.config import ProjectConfig
from kryptonite.data.audio_loader import AudioLoadRequest

EXPORT_BOUNDARY_FORMAT_VERSION = "kryptonite.serve.export_boundary.v1"
SUPPORTED_EXPORT_BOUNDARY_MODES = frozenset({"encoder_only"})
_DYNAMIC_AXIS = "dynamic"


@dataclass(frozen=True, slots=True)
class TensorAxisSpec:
    name: str
    size: int | str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "size": self.size}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TensorAxisSpec:
        name = payload.get("name")
        size = payload.get("size")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tensor axis metadata must define a non-empty string name.")
        if not isinstance(size, (int, str)):
            raise ValueError(f"Tensor axis {name!r} must define an int or string size.")
        return cls(name=name, size=size)


@dataclass(frozen=True, slots=True)
class TensorContract:
    name: str
    layout: str
    dtype: str
    semantic: str
    axes: tuple[TensorAxisSpec, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "layout": self.layout,
            "dtype": self.dtype,
            "semantic": self.semantic,
            "axes": [axis.to_dict() for axis in self.axes],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TensorContract:
        name = payload.get("name")
        layout = payload.get("layout")
        dtype = payload.get("dtype")
        semantic = payload.get("semantic")
        raw_axes = payload.get("axes")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tensor metadata must define a non-empty `name`.")
        if not isinstance(layout, str) or not layout.strip():
            raise ValueError(f"Tensor {name!r} must define a non-empty `layout`.")
        if not isinstance(dtype, str) or not dtype.strip():
            raise ValueError(f"Tensor {name!r} must define a non-empty `dtype`.")
        if not isinstance(semantic, str) or not semantic.strip():
            raise ValueError(f"Tensor {name!r} must define a non-empty `semantic`.")
        if not isinstance(raw_axes, list) or not raw_axes:
            raise ValueError(f"Tensor {name!r} must define a non-empty `axes` list.")
        axes = []
        for item in raw_axes:
            if not isinstance(item, Mapping):
                raise ValueError(f"Tensor {name!r} axes must contain object entries.")
            axes.append(TensorAxisSpec.from_dict(cast(Mapping[str, object], item)))
        return cls(
            name=name,
            layout=layout,
            dtype=dtype,
            semantic=semantic,
            axes=tuple(axes),
        )


@dataclass(frozen=True, slots=True)
class ExportBoundaryContract:
    format_version: str
    boundary: str
    export_profile: str
    dynamic_time_axis: bool
    frontend_location: str
    inferencer_backend: str
    embedding_stage: str
    embedding_mode: str | None
    input_tensor: TensorContract
    output_tensor: TensorContract
    runtime_frontend: dict[str, Any]
    runtime_pre_engine_steps: tuple[str, ...]
    runtime_post_engine_steps: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "boundary": self.boundary,
            "export_profile": self.export_profile,
            "dynamic_time_axis": self.dynamic_time_axis,
            "frontend_location": self.frontend_location,
            "inferencer_backend": self.inferencer_backend,
            "embedding_stage": self.embedding_stage,
            "embedding_mode": self.embedding_mode,
            "input_tensor": self.input_tensor.to_dict(),
            "output_tensor": self.output_tensor.to_dict(),
            "runtime_frontend": json.loads(json.dumps(self.runtime_frontend, sort_keys=True)),
            "runtime_pre_engine_steps": list(self.runtime_pre_engine_steps),
            "runtime_post_engine_steps": list(self.runtime_post_engine_steps),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExportBoundaryContract:
        format_version = payload.get("format_version")
        boundary = payload.get("boundary")
        export_profile = payload.get("export_profile")
        dynamic_time_axis = payload.get("dynamic_time_axis")
        frontend_location = payload.get("frontend_location")
        inferencer_backend = payload.get("inferencer_backend")
        embedding_stage = payload.get("embedding_stage")
        embedding_mode = payload.get("embedding_mode")
        runtime_frontend = payload.get("runtime_frontend")
        pre_engine_steps = payload.get("runtime_pre_engine_steps")
        post_engine_steps = payload.get("runtime_post_engine_steps")
        if format_version != EXPORT_BOUNDARY_FORMAT_VERSION:
            raise ValueError(
                "Unsupported export boundary format version: "
                f"{format_version!r}; expected {EXPORT_BOUNDARY_FORMAT_VERSION!r}."
            )
        normalized_boundary = _normalize_export_boundary_mode(boundary)
        if not isinstance(export_profile, str) or not export_profile.strip():
            raise ValueError("Export boundary metadata must define `export_profile`.")
        if not isinstance(dynamic_time_axis, bool):
            raise ValueError("Export boundary metadata must define boolean `dynamic_time_axis`.")
        if frontend_location != "runtime":
            raise ValueError(
                'Export boundary metadata currently supports only `frontend_location="runtime"`.'
            )
        if not isinstance(inferencer_backend, str) or not inferencer_backend.strip():
            raise ValueError("Export boundary metadata must define `inferencer_backend`.")
        if not isinstance(embedding_stage, str) or not embedding_stage.strip():
            raise ValueError("Export boundary metadata must define `embedding_stage`.")
        if embedding_mode is not None and not isinstance(embedding_mode, str):
            raise ValueError("Export boundary metadata `embedding_mode` must be a string or null.")
        if not isinstance(runtime_frontend, dict):
            raise ValueError("Export boundary metadata must define `runtime_frontend` object.")
        if not isinstance(pre_engine_steps, list) or not pre_engine_steps:
            raise ValueError(
                "Export boundary metadata must define non-empty `runtime_pre_engine_steps`."
            )
        if not isinstance(post_engine_steps, list) or not post_engine_steps:
            raise ValueError(
                "Export boundary metadata must define non-empty `runtime_post_engine_steps`."
            )
        input_tensor_payload = payload.get("input_tensor")
        output_tensor_payload = payload.get("output_tensor")
        if not isinstance(input_tensor_payload, Mapping):
            raise ValueError("Export boundary metadata must define `input_tensor`.")
        if not isinstance(output_tensor_payload, Mapping):
            raise ValueError("Export boundary metadata must define `output_tensor`.")
        return cls(
            format_version=cast(str, format_version),
            boundary=normalized_boundary,
            export_profile=export_profile,
            dynamic_time_axis=dynamic_time_axis,
            frontend_location=cast(str, frontend_location),
            inferencer_backend=inferencer_backend,
            embedding_stage=embedding_stage,
            embedding_mode=embedding_mode,
            input_tensor=TensorContract.from_dict(cast(Mapping[str, object], input_tensor_payload)),
            output_tensor=TensorContract.from_dict(
                cast(Mapping[str, object], output_tensor_payload)
            ),
            runtime_frontend=_copy_runtime_frontend(cast(Mapping[str, object], runtime_frontend)),
            runtime_pre_engine_steps=tuple(_coerce_step_list(cast(list[object], pre_engine_steps))),
            runtime_post_engine_steps=tuple(
                _coerce_step_list(cast(list[object], post_engine_steps))
            ),
        )

    def summary_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "boundary": self.boundary,
            "frontend_location": self.frontend_location,
            "dynamic_time_axis": self.dynamic_time_axis,
            "inferencer_backend": self.inferencer_backend,
            "embedding_stage": self.embedding_stage,
            "embedding_mode": self.embedding_mode,
            "input_name": self.input_tensor.name,
            "input_layout": self.input_tensor.layout,
            "input_dtype": self.input_tensor.dtype,
            "output_name": self.output_tensor.name,
            "output_layout": self.output_tensor.layout,
            "output_dtype": self.output_tensor.dtype,
            "pre_engine_steps": list(self.runtime_pre_engine_steps),
            "post_engine_steps": list(self.runtime_post_engine_steps),
        }


def build_export_boundary_contract(
    *,
    config: ProjectConfig,
    inferencer_backend: str = "feature_statistics",
    embedding_stage: str = "demo",
    embedding_mode: str | None = "mean_std",
    embedding_dim: int | None = None,
) -> ExportBoundaryContract:
    boundary = _normalize_export_boundary_mode(config.export.boundary)
    resolved_embedding_dim = _resolve_embedding_dim(
        config=config,
        inferencer_backend=inferencer_backend,
        embedding_mode=embedding_mode,
        explicit_embedding_dim=embedding_dim,
    )
    return ExportBoundaryContract(
        format_version=EXPORT_BOUNDARY_FORMAT_VERSION,
        boundary=boundary,
        export_profile=config.export.profile,
        dynamic_time_axis=config.export.dynamic_axes,
        frontend_location="runtime",
        inferencer_backend=inferencer_backend,
        embedding_stage=embedding_stage,
        embedding_mode=embedding_mode,
        input_tensor=TensorContract(
            name=config.export.input_name,
            layout="BTF",
            dtype=config.features.output_dtype,
            semantic="log_mel_fbank_frames",
            axes=(
                TensorAxisSpec(name="batch", size=_DYNAMIC_AXIS),
                TensorAxisSpec(
                    name="frames",
                    size=_DYNAMIC_AXIS if config.export.dynamic_axes else "profile_bound",
                ),
                TensorAxisSpec(name="mel_bins", size=config.features.num_mel_bins),
            ),
        ),
        output_tensor=TensorContract(
            name=config.export.output_name,
            layout="BE",
            dtype="float32",
            semantic="speaker_embedding",
            axes=(
                TensorAxisSpec(name="batch", size=_DYNAMIC_AXIS),
                TensorAxisSpec(name="embedding_dim", size=resolved_embedding_dim),
            ),
        ),
        runtime_frontend=_build_runtime_frontend_contract(config),
        runtime_pre_engine_steps=_runtime_pre_engine_steps(config),
        runtime_post_engine_steps=(
            "pool_chunk_embeddings",
            "average_enrollment_embeddings",
            "normalize_embeddings_for_scoring",
            "score_and_threshold_outside_engine",
        ),
    )


def build_model_bundle_metadata(
    *,
    config: ProjectConfig,
    model_file: str,
    enrollment_cache_compatibility_id: str,
    description: str,
    inferencer_backend: str = "feature_statistics",
    embedding_stage: str = "demo",
    embedding_mode: str | None = "mean_std",
    embedding_dim: int | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    contract = build_export_boundary_contract(
        config=config,
        inferencer_backend=inferencer_backend,
        embedding_stage=embedding_stage,
        embedding_mode=embedding_mode,
        embedding_dim=embedding_dim,
    )
    metadata: dict[str, Any] = {
        "model_file": model_file,
        "input_name": contract.input_tensor.name,
        "output_name": contract.output_tensor.name,
        "inferencer_backend": inferencer_backend,
        "embedding_stage": embedding_stage,
        "embedding_mode": embedding_mode,
        "enrollment_cache_compatibility_id": enrollment_cache_compatibility_id,
        "description": description,
        "export_boundary": contract.to_dict(),
    }
    if extra_metadata is not None:
        metadata.update(dict(extra_metadata))
    return metadata


def load_export_boundary_from_model_metadata(
    model_metadata: Mapping[str, object],
) -> ExportBoundaryContract:
    raw_payload = model_metadata.get("export_boundary")
    if not isinstance(raw_payload, Mapping):
        raise ValueError(
            "Model bundle metadata is missing the `export_boundary` contract. "
            "Regenerate the model bundle metadata before using it in runtime or deploy flows."
        )
    contract = ExportBoundaryContract.from_dict(cast(Mapping[str, object], raw_payload))
    input_name = model_metadata.get("input_name")
    output_name = model_metadata.get("output_name")
    if input_name is not None and input_name != contract.input_tensor.name:
        raise ValueError(
            "Model bundle metadata input_name does not match export_boundary.input_tensor.name: "
            f"{input_name!r} != {contract.input_tensor.name!r}."
        )
    if output_name is not None and output_name != contract.output_tensor.name:
        raise ValueError(
            "Model bundle metadata output_name does not match export_boundary.output_tensor.name: "
            f"{output_name!r} != {contract.output_tensor.name!r}."
        )
    return contract


def validate_runtime_frontend_against_boundary(
    *,
    config: ProjectConfig,
    contract: ExportBoundaryContract,
) -> None:
    mismatches: list[str] = []
    if contract.boundary != _normalize_export_boundary_mode(config.export.boundary):
        mismatches.append(
            "boundary mode mismatch: "
            f"model_bundle={contract.boundary!r}, config={config.export.boundary!r}"
        )
    if contract.input_tensor.name != config.export.input_name:
        mismatches.append(
            "input name mismatch: "
            f"model_bundle={contract.input_tensor.name!r}, config={config.export.input_name!r}"
        )
    if contract.output_tensor.name != config.export.output_name:
        mismatches.append(
            "output name mismatch: "
            f"model_bundle={contract.output_tensor.name!r}, config={config.export.output_name!r}"
        )
    expected_frontend = _build_runtime_frontend_contract(config)
    mismatches.extend(
        _collect_dict_mismatches(
            section="runtime_frontend.audio_load_request",
            expected=expected_frontend["audio_load_request"],
            actual=contract.runtime_frontend.get("audio_load_request"),
        )
    )
    mismatches.extend(
        _collect_dict_mismatches(
            section="runtime_frontend.features",
            expected=expected_frontend["features"],
            actual=contract.runtime_frontend.get("features"),
        )
    )
    mismatches.extend(
        _collect_dict_mismatches(
            section="runtime_frontend.chunking",
            expected=expected_frontend["chunking"],
            actual=contract.runtime_frontend.get("chunking"),
        )
    )
    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(f"Model bundle export boundary mismatch: {joined}")


def render_export_boundary_markdown(contract: ExportBoundaryContract) -> str:
    audio_request = contract.runtime_frontend["audio_load_request"]
    features = contract.runtime_frontend["features"]
    chunking = contract.runtime_frontend["chunking"]
    return "\n".join(
        [
            "# Export Boundary",
            "",
            f"- format version: `{contract.format_version}`",
            f"- boundary mode: `{contract.boundary}`",
            f"- export profile: `{contract.export_profile}`",
            f"- frontend location: `{contract.frontend_location}`",
            f"- inferencer backend: `{contract.inferencer_backend}`",
            f"- embedding stage: `{contract.embedding_stage}`",
            f"- embedding mode: `{contract.embedding_mode}`",
            f"- input tensor: `{_render_tensor_signature(contract.input_tensor)}`",
            f"- output tensor: `{_render_tensor_signature(contract.output_tensor)}`",
            "",
            "## Runtime Responsibilities",
            "",
            f"- pre-engine: `{', '.join(contract.runtime_pre_engine_steps)}`",
            f"- post-engine: `{', '.join(contract.runtime_post_engine_steps)}`",
            "",
            "## Frontend Config",
            "",
            f"- sample rate: `{audio_request['target_sample_rate_hz']} Hz`",
            f"- channels: `{audio_request['target_channels']}`",
            f"- loudness mode: `{audio_request['loudness_mode']}`",
            f"- VAD mode/backend: `{audio_request['vad_mode']}` / `{audio_request['vad_backend']}`",
            f"- Fbank bins: `{features['num_mel_bins']}`",
            f"- frame length / shift: `{features['frame_length_ms']} ms` / "
            f"`{features['frame_shift_ms']} ms`",
            f"- CMVN: `{features['cmvn_mode']}`",
            f"- demo chunk / overlap / pooling: `{chunking['demo_chunk_seconds']} s` / "
            f"`{chunking['demo_chunk_overlap_seconds']} s` / `{chunking['demo_pooling']}`",
            f"- eval chunk / overlap / pooling: `{chunking['eval_chunk_seconds']} s` / "
            f"`{chunking['eval_chunk_overlap_seconds']} s` / `{chunking['eval_pooling']}`",
        ]
    )


def _build_runtime_frontend_contract(config: ProjectConfig) -> dict[str, Any]:
    return {
        "audio_load_request": asdict(
            AudioLoadRequest.from_config(config.normalization, vad=config.vad)
        ),
        "features": asdict(config.features),
        "chunking": asdict(config.chunking),
    }


def _runtime_pre_engine_steps(config: ProjectConfig) -> tuple[str, ...]:
    audio_request = AudioLoadRequest.from_config(config.normalization, vad=config.vad)
    loudness_step = (
        "apply_loudness_normalization"
        if audio_request.loudness_mode != "none"
        else "skip_loudness_normalization"
    )
    vad_step = f"apply_vad_{audio_request.vad_mode}"
    return (
        "decode_audio",
        "fold_channels_before_engine",
        "resample_before_engine",
        loudness_step,
        vad_step,
        "chunk_waveform_before_engine",
        "extract_fbank_before_engine",
    )


def _resolve_embedding_dim(
    *,
    config: ProjectConfig,
    inferencer_backend: str,
    embedding_mode: str | None,
    explicit_embedding_dim: int | None,
) -> int | str:
    if explicit_embedding_dim is not None:
        return explicit_embedding_dim
    if inferencer_backend != "feature_statistics":
        return _DYNAMIC_AXIS
    normalized_mode = "mean_std" if embedding_mode is None else embedding_mode.lower()
    if normalized_mode == "mean":
        return config.features.num_mel_bins
    return config.features.num_mel_bins * 2


def _render_tensor_signature(tensor: TensorContract) -> str:
    axes = ", ".join(f"{axis.name}={axis.size}" for axis in tensor.axes)
    return f"{tensor.name} ({tensor.layout}, {tensor.dtype}; {axes})"


def _normalize_export_boundary_mode(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Export boundary mode must be a non-empty string.")
    normalized = value.lower()
    if normalized not in SUPPORTED_EXPORT_BOUNDARY_MODES:
        raise ValueError(
            "Unsupported export boundary mode "
            f"{value!r}; expected one of {sorted(SUPPORTED_EXPORT_BOUNDARY_MODES)}."
        )
    return normalized


def _coerce_step_list(values: list[object]) -> list[str]:
    steps: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Export boundary step lists must contain non-empty strings.")
        steps.append(value)
    return steps


def _copy_runtime_frontend(payload: Mapping[str, object]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(payload), sort_keys=True))
    except TypeError as exc:
        raise ValueError(
            "Export boundary runtime_frontend payload must be JSON-serializable."
        ) from exc


def _collect_dict_mismatches(
    *,
    section: str,
    expected: Mapping[str, Any],
    actual: object,
) -> list[str]:
    if not isinstance(actual, Mapping):
        return [f"{section} missing or malformed"]
    mismatches: list[str] = []
    actual_dict = dict(actual)
    for key in sorted(set(expected) | set(actual_dict)):
        expected_value = expected.get(key)
        actual_value = actual_dict.get(key)
        if expected_value != actual_value:
            mismatches.append(
                f"{section}.{key}: model_bundle={actual_value!r}, config={expected_value!r}"
            )
    return mismatches


__all__ = [
    "EXPORT_BOUNDARY_FORMAT_VERSION",
    "SUPPORTED_EXPORT_BOUNDARY_MODES",
    "ExportBoundaryContract",
    "TensorAxisSpec",
    "TensorContract",
    "build_export_boundary_contract",
    "build_model_bundle_metadata",
    "load_export_boundary_from_model_metadata",
    "render_export_boundary_markdown",
    "validate_runtime_frontend_against_boundary",
]
