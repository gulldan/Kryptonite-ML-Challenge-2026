"""Typed config loader for TensorRT FP16 engine builds."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TensorRTFP16ArtifactsConfig:
    model_bundle_metadata_path: str
    engine_output_path: str

    def __post_init__(self) -> None:
        if not self.model_bundle_metadata_path.strip():
            raise ValueError("artifacts.model_bundle_metadata_path must be a non-empty string.")
        if not self.engine_output_path.strip():
            raise ValueError("artifacts.engine_output_path must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class TensorRTFP16BuildProfileConfig:
    profile_id: str = "default"
    min_batch_size: int = 1
    opt_batch_size: int = 4
    max_batch_size: int = 8
    min_frame_count: int = 80
    opt_frame_count: int = 200
    max_frame_count: int = 800

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("build.profiles[].profile_id must be a non-empty string.")
        for field_name in (
            "min_batch_size",
            "opt_batch_size",
            "max_batch_size",
            "min_frame_count",
            "opt_frame_count",
            "max_frame_count",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if not self.min_batch_size <= self.opt_batch_size <= self.max_batch_size:
            raise ValueError("batch sizes must satisfy min <= opt <= max.")
        if not self.min_frame_count <= self.opt_frame_count <= self.max_frame_count:
            raise ValueError("frame counts must satisfy min <= opt <= max.")

    def covers(self, *, batch_size: int, frame_count: int) -> bool:
        return (
            self.min_batch_size <= batch_size <= self.max_batch_size
            and self.min_frame_count <= frame_count <= self.max_frame_count
        )


@dataclass(frozen=True, slots=True)
class TensorRTFP16BuildConfig:
    workspace_size_mib: int = 2_048
    profiles: tuple[TensorRTFP16BuildProfileConfig, ...] = (TensorRTFP16BuildProfileConfig(),)
    promote_validated_backend: bool = True
    require_onnxruntime_parity: bool = True

    def __post_init__(self) -> None:
        if self.workspace_size_mib <= 0:
            raise ValueError("workspace_size_mib must be positive.")
        if not self.profiles:
            raise ValueError("build.profiles must define at least one optimization profile.")
        profile_ids = [profile.profile_id for profile in self.profiles]
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("build.profiles must use unique profile_id values.")

    @property
    def min_batch_size(self) -> int:
        return min(profile.min_batch_size for profile in self.profiles)

    @property
    def max_batch_size(self) -> int:
        return max(profile.max_batch_size for profile in self.profiles)

    @property
    def min_frame_count(self) -> int:
        return min(profile.min_frame_count for profile in self.profiles)

    @property
    def max_frame_count(self) -> int:
        return max(profile.max_frame_count for profile in self.profiles)


@dataclass(frozen=True, slots=True)
class TensorRTFP16SampleConfig:
    sample_id: str
    batch_size: int
    frame_count: int

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("samples[].sample_id must be a non-empty string.")
        if self.batch_size <= 0:
            raise ValueError("samples[].batch_size must be positive.")
        if self.frame_count <= 0:
            raise ValueError("samples[].frame_count must be positive.")


@dataclass(frozen=True, slots=True)
class TensorRTFP16EvaluationConfig:
    seed: int
    warmup_iterations: int
    benchmark_iterations: int
    max_mean_abs_diff: float
    max_cosine_distance: float
    min_speedup_ratio: float
    samples: tuple[TensorRTFP16SampleConfig, ...]

    def __post_init__(self) -> None:
        if self.warmup_iterations < 0:
            raise ValueError("evaluation.warmup_iterations must be non-negative.")
        if self.benchmark_iterations <= 0:
            raise ValueError("evaluation.benchmark_iterations must be positive.")
        if self.max_mean_abs_diff < 0.0:
            raise ValueError("evaluation.max_mean_abs_diff must be non-negative.")
        if self.max_cosine_distance < 0.0:
            raise ValueError("evaluation.max_cosine_distance must be non-negative.")
        if self.min_speedup_ratio <= 0.0:
            raise ValueError("evaluation.min_speedup_ratio must be positive.")
        if not self.samples:
            raise ValueError("evaluation.samples must define at least one sample.")


@dataclass(frozen=True, slots=True)
class TensorRTFP16Config:
    title: str
    report_id: str
    summary: str
    project_root: str
    output_root: str
    artifacts: TensorRTFP16ArtifactsConfig
    build: TensorRTFP16BuildConfig
    evaluation: TensorRTFP16EvaluationConfig
    validation_commands: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must be a non-empty string.")
        if not self.report_id.strip():
            raise ValueError("report_id must be a non-empty string.")
        if not self.project_root.strip():
            raise ValueError("project_root must be a non-empty string.")
        if not self.output_root.strip():
            raise ValueError("output_root must be a non-empty string.")
        for sample in self.evaluation.samples:
            if sample.batch_size < self.build.min_batch_size:
                raise ValueError(
                    f"Sample {sample.sample_id!r} batch size is smaller than build.min_batch_size."
                )
            if sample.batch_size > self.build.max_batch_size:
                raise ValueError(
                    f"Sample {sample.sample_id!r} batch size exceeds build.max_batch_size."
                )
            if sample.frame_count < self.build.min_frame_count:
                raise ValueError(
                    f"Sample {sample.sample_id!r} frame count is smaller than "
                    "build.min_frame_count."
                )
            if sample.frame_count > self.build.max_frame_count:
                raise ValueError(
                    f"Sample {sample.sample_id!r} frame count exceeds build.max_frame_count."
                )
            if not any(
                profile.covers(
                    batch_size=sample.batch_size,
                    frame_count=sample.frame_count,
                )
                for profile in self.build.profiles
            ):
                raise ValueError(
                    f"Sample {sample.sample_id!r} is not covered by any build.profiles entry."
                )


def load_tensorrt_fp16_config(*, config_path: Path | str) -> TensorRTFP16Config:
    path = Path(config_path)
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a TOML object in {path}.")

    artifacts = _coerce_mapping(payload.get("artifacts"), field_name="artifacts")
    build = _coerce_mapping(payload.get("build"), field_name="build")
    evaluation = _coerce_mapping(payload.get("evaluation"), field_name="evaluation")

    config = TensorRTFP16Config(
        title=_coerce_string(payload.get("title"), field_name="title"),
        report_id=_coerce_string(payload.get("report_id"), field_name="report_id"),
        summary=_coerce_string(payload.get("summary"), field_name="summary"),
        project_root=_coerce_optional_string(payload.get("project_root")) or ".",
        output_root=_coerce_string(payload.get("output_root"), field_name="output_root"),
        artifacts=TensorRTFP16ArtifactsConfig(
            model_bundle_metadata_path=_coerce_string(
                artifacts.get("model_bundle_metadata_path"),
                field_name="artifacts.model_bundle_metadata_path",
            ),
            engine_output_path=_coerce_string(
                artifacts.get("engine_output_path"),
                field_name="artifacts.engine_output_path",
            ),
        ),
        build=TensorRTFP16BuildConfig(
            workspace_size_mib=_coerce_positive_int(
                build.get("workspace_size_mib", 2_048),
                field_name="build.workspace_size_mib",
            ),
            profiles=_parse_build_profiles(build),
            promote_validated_backend=_coerce_bool(
                build.get("promote_validated_backend", True),
                field_name="build.promote_validated_backend",
            ),
            require_onnxruntime_parity=_coerce_bool(
                build.get("require_onnxruntime_parity", True),
                field_name="build.require_onnxruntime_parity",
            ),
        ),
        evaluation=TensorRTFP16EvaluationConfig(
            seed=_coerce_int(evaluation.get("seed", 0), field_name="evaluation.seed"),
            warmup_iterations=_coerce_non_negative_int(
                evaluation.get("warmup_iterations", 10),
                field_name="evaluation.warmup_iterations",
            ),
            benchmark_iterations=_coerce_positive_int(
                evaluation.get("benchmark_iterations", 50),
                field_name="evaluation.benchmark_iterations",
            ),
            max_mean_abs_diff=_coerce_non_negative_float(
                evaluation.get("max_mean_abs_diff", 0.01),
                field_name="evaluation.max_mean_abs_diff",
            ),
            max_cosine_distance=_coerce_non_negative_float(
                evaluation.get("max_cosine_distance", 0.0005),
                field_name="evaluation.max_cosine_distance",
            ),
            min_speedup_ratio=_coerce_positive_float(
                evaluation.get("min_speedup_ratio", 1.0),
                field_name="evaluation.min_speedup_ratio",
            ),
            samples=_parse_samples(evaluation.get("samples")),
        ),
        validation_commands=_coerce_optional_string_tuple(payload.get("validation_commands")),
        notes=_coerce_optional_string_tuple(payload.get("notes")),
    )
    return config


def _parse_build_profiles(raw_build: dict[str, Any]) -> tuple[TensorRTFP16BuildProfileConfig, ...]:
    raw_profiles = raw_build.get("profiles")
    if raw_profiles is None:
        return (
            TensorRTFP16BuildProfileConfig(
                profile_id="default",
                min_batch_size=_coerce_positive_int(
                    raw_build.get("min_batch_size", 1),
                    field_name="build.min_batch_size",
                ),
                opt_batch_size=_coerce_positive_int(
                    raw_build.get("opt_batch_size", 4),
                    field_name="build.opt_batch_size",
                ),
                max_batch_size=_coerce_positive_int(
                    raw_build.get("max_batch_size", 8),
                    field_name="build.max_batch_size",
                ),
                min_frame_count=_coerce_positive_int(
                    raw_build.get("min_frame_count", 80),
                    field_name="build.min_frame_count",
                ),
                opt_frame_count=_coerce_positive_int(
                    raw_build.get("opt_frame_count", 200),
                    field_name="build.opt_frame_count",
                ),
                max_frame_count=_coerce_positive_int(
                    raw_build.get("max_frame_count", 800),
                    field_name="build.max_frame_count",
                ),
            ),
        )
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("build.profiles must be a non-empty array of tables.")

    profiles: list[TensorRTFP16BuildProfileConfig] = []
    for index, item in enumerate(raw_profiles):
        payload = _coerce_mapping(item, field_name=f"build.profiles[{index}]")
        profiles.append(
            TensorRTFP16BuildProfileConfig(
                profile_id=_coerce_string(
                    payload.get("profile_id"),
                    field_name=f"build.profiles[{index}].profile_id",
                ),
                min_batch_size=_coerce_positive_int(
                    payload.get("min_batch_size", 1),
                    field_name=f"build.profiles[{index}].min_batch_size",
                ),
                opt_batch_size=_coerce_positive_int(
                    payload.get("opt_batch_size", 4),
                    field_name=f"build.profiles[{index}].opt_batch_size",
                ),
                max_batch_size=_coerce_positive_int(
                    payload.get("max_batch_size", 8),
                    field_name=f"build.profiles[{index}].max_batch_size",
                ),
                min_frame_count=_coerce_positive_int(
                    payload.get("min_frame_count", 80),
                    field_name=f"build.profiles[{index}].min_frame_count",
                ),
                opt_frame_count=_coerce_positive_int(
                    payload.get("opt_frame_count", 200),
                    field_name=f"build.profiles[{index}].opt_frame_count",
                ),
                max_frame_count=_coerce_positive_int(
                    payload.get("max_frame_count", 800),
                    field_name=f"build.profiles[{index}].max_frame_count",
                ),
            )
        )
    return tuple(profiles)


def _parse_samples(raw: object) -> tuple[TensorRTFP16SampleConfig, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("evaluation.samples must be a non-empty array of tables.")
    samples: list[TensorRTFP16SampleConfig] = []
    for index, item in enumerate(raw):
        sample_payload = _coerce_mapping(item, field_name=f"evaluation.samples[{index}]")
        samples.append(
            TensorRTFP16SampleConfig(
                sample_id=_coerce_string(
                    sample_payload.get("sample_id"),
                    field_name=f"evaluation.samples[{index}].sample_id",
                ),
                batch_size=_coerce_positive_int(
                    sample_payload.get("batch_size"),
                    field_name=f"evaluation.samples[{index}].batch_size",
                ),
                frame_count=_coerce_positive_int(
                    sample_payload.get("frame_count"),
                    field_name=f"evaluation.samples[{index}].frame_count",
                ),
            )
        )
    return tuple(samples)


def _coerce_mapping(raw: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a TOML table.")
    return {str(key): value for key, value in raw.items()}


def _coerce_optional_string_tuple(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("Expected a list of strings.")
    values: list[str] = []
    for item in raw:
        values.append(_coerce_string(item, field_name="list entry"))
    return tuple(values)


def _coerce_string(raw: object, *, field_name: str) -> str:
    value = _coerce_optional_string(raw)
    if value is None:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _coerce_optional_string(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    return value or None


def _coerce_bool(raw: object, *, field_name: str) -> bool:
    if not isinstance(raw, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return raw


def _coerce_int(raw: object, *, field_name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{field_name} must be an integer.")
    return raw


def _coerce_positive_int(raw: object, *, field_name: str) -> int:
    value = _coerce_int(raw, field_name=field_name)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return value


def _coerce_non_negative_int(raw: object, *, field_name: str) -> int:
    value = _coerce_int(raw, field_name=field_name)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return value


def _coerce_positive_float(raw: object, *, field_name: str) -> float:
    value = _coerce_float(raw, field_name=field_name)
    if value <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return value


def _coerce_non_negative_float(raw: object, *, field_name: str) -> float:
    value = _coerce_float(raw, field_name=field_name)
    if value < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")
    return value


def _coerce_float(raw: object, *, field_name: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    return float(raw)


__all__ = [
    "TensorRTFP16ArtifactsConfig",
    "TensorRTFP16BuildConfig",
    "TensorRTFP16BuildProfileConfig",
    "TensorRTFP16Config",
    "TensorRTFP16EvaluationConfig",
    "TensorRTFP16SampleConfig",
    "load_tensorrt_fp16_config",
]
