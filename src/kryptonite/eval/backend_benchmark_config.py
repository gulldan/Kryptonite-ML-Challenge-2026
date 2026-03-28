"""Typed config loader for reproducible backend benchmark reports."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

_SUPPORTED_BACKEND_BENCHMARK_BACKENDS = frozenset({"torch", "onnxruntime", "tensorrt"})
_SUPPORTED_BACKEND_BENCHMARK_DEVICES = frozenset({"auto", "cpu", "cuda"})
_SUPPORTED_ONNXRUNTIME_PROVIDERS = frozenset({"auto", "cpu", "cuda"})


@dataclass(frozen=True, slots=True)
class BackendBenchmarkArtifactsConfig:
    model_bundle_metadata_path: str
    tensorrt_report_path: str
    source_checkpoint_path_override: str | None = None
    onnx_model_path_override: str | None = None
    tensorrt_engine_path_override: str | None = None

    def __post_init__(self) -> None:
        if not self.model_bundle_metadata_path.strip():
            raise ValueError("artifacts.model_bundle_metadata_path must be a non-empty string.")
        if not self.tensorrt_report_path.strip():
            raise ValueError("artifacts.tensorrt_report_path must be a non-empty string.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BackendBenchmarkEvaluationConfig:
    seed: int
    device: str
    onnxruntime_provider: str
    warmup_iterations: int
    benchmark_iterations: int
    backends: tuple[str, ...]
    max_mean_abs_diff: float
    max_cosine_distance: float

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("evaluation.seed must be non-negative.")
        if self.device not in _SUPPORTED_BACKEND_BENCHMARK_DEVICES:
            raise ValueError(
                f"evaluation.device must be one of {sorted(_SUPPORTED_BACKEND_BENCHMARK_DEVICES)}."
            )
        if self.onnxruntime_provider not in _SUPPORTED_ONNXRUNTIME_PROVIDERS:
            raise ValueError(
                "evaluation.onnxruntime_provider must be one of "
                f"{sorted(_SUPPORTED_ONNXRUNTIME_PROVIDERS)}."
            )
        if self.warmup_iterations < 0:
            raise ValueError("evaluation.warmup_iterations must be non-negative.")
        if self.benchmark_iterations <= 0:
            raise ValueError("evaluation.benchmark_iterations must be positive.")
        if self.max_mean_abs_diff < 0.0:
            raise ValueError("evaluation.max_mean_abs_diff must be non-negative.")
        if self.max_cosine_distance < 0.0:
            raise ValueError("evaluation.max_cosine_distance must be non-negative.")
        if not self.backends:
            raise ValueError("evaluation.backends must not be empty.")
        if "torch" not in self.backends:
            raise ValueError("evaluation.backends must include `torch` as the reference backend.")
        if len(set(self.backends)) != len(self.backends):
            raise ValueError("evaluation.backends must not contain duplicate backend names.")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["backends"] = list(self.backends)
        return payload


@dataclass(frozen=True, slots=True)
class BackendBenchmarkWorkloadConfig:
    workload_id: str
    batch_size: int
    frame_count: int
    description: str

    def __post_init__(self) -> None:
        if not self.workload_id.strip():
            raise ValueError("workloads[].id must be a non-empty string.")
        if self.batch_size <= 0:
            raise ValueError("workloads[].batch_size must be positive.")
        if self.frame_count <= 0:
            raise ValueError("workloads[].frame_count must be positive.")
        if not self.description.strip():
            raise ValueError("workloads[].description must be a non-empty string.")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.workload_id,
            "batch_size": self.batch_size,
            "frame_count": self.frame_count,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class BackendBenchmarkConfig:
    title: str
    report_id: str
    summary: str
    project_root: str
    output_root: str
    artifacts: BackendBenchmarkArtifactsConfig
    evaluation: BackendBenchmarkEvaluationConfig
    workloads: tuple[BackendBenchmarkWorkloadConfig, ...]
    validation_commands: tuple[str, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must be a non-empty string.")
        if not self.report_id.strip():
            raise ValueError("report_id must be a non-empty string.")
        if not self.project_root.strip():
            raise ValueError("project_root must be a non-empty string.")
        if not self.output_root.strip():
            raise ValueError("output_root must be a non-empty string.")
        if not self.workloads:
            raise ValueError("At least one workload must be configured.")
        seen_ids: set[str] = set()
        has_batch_size_one = False
        has_batched = False
        for workload in self.workloads:
            if workload.workload_id in seen_ids:
                raise ValueError(f"Duplicate workloads[].id value: {workload.workload_id!r}.")
            seen_ids.add(workload.workload_id)
            has_batch_size_one = has_batch_size_one or workload.batch_size == 1
            has_batched = has_batched or workload.batch_size > 1
        if not has_batch_size_one:
            raise ValueError("At least one workload must use batch_size=1.")
        if not has_batched:
            raise ValueError("At least one workload must use batch_size>1.")

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "report_id": self.report_id,
            "summary": self.summary,
            "project_root": self.project_root,
            "output_root": self.output_root,
            "artifacts": self.artifacts.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "workloads": [workload.to_dict() for workload in self.workloads],
            "validation_commands": list(self.validation_commands),
            "notes": list(self.notes),
        }


def load_backend_benchmark_config(*, config_path: Path | str) -> BackendBenchmarkConfig:
    payload = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    artifacts = _coerce_table(payload.get("artifacts"), "artifacts")
    evaluation = _coerce_table(payload.get("evaluation"), "evaluation")
    raw_workloads = payload.get("workloads")
    if not isinstance(raw_workloads, list) or not raw_workloads:
        raise ValueError("workloads must be a non-empty array of tables.")

    workloads: list[BackendBenchmarkWorkloadConfig] = []
    for index, raw_workload in enumerate(raw_workloads):
        workload = _coerce_table(raw_workload, f"workloads[{index}]")
        workload_id = str(workload.get("id", "")).strip()
        batch_size = _coerce_required_int(
            workload.get("batch_size"),
            f"workloads[{index}].batch_size",
        )
        frame_count = _coerce_required_int(
            workload.get("frame_count"),
            f"workloads[{index}].frame_count",
        )
        description = str(workload.get("description", "")).strip() or (
            f"batch={batch_size}, frames={frame_count}"
        )
        workloads.append(
            BackendBenchmarkWorkloadConfig(
                workload_id=workload_id,
                batch_size=batch_size,
                frame_count=frame_count,
                description=description,
            )
        )

    return BackendBenchmarkConfig(
        title=str(payload.get("title", "")).strip(),
        report_id=str(payload.get("report_id", "")).strip(),
        summary=str(payload.get("summary", "")).strip(),
        project_root=str(payload.get("project_root", ".")).strip() or ".",
        output_root=str(payload.get("output_root", "")).strip(),
        artifacts=BackendBenchmarkArtifactsConfig(
            model_bundle_metadata_path=str(artifacts.get("model_bundle_metadata_path", "")).strip(),
            tensorrt_report_path=str(artifacts.get("tensorrt_report_path", "")).strip(),
            source_checkpoint_path_override=_coerce_optional_string(
                artifacts.get("source_checkpoint_path_override")
            ),
            onnx_model_path_override=_coerce_optional_string(
                artifacts.get("onnx_model_path_override")
            ),
            tensorrt_engine_path_override=_coerce_optional_string(
                artifacts.get("tensorrt_engine_path_override")
            ),
        ),
        evaluation=BackendBenchmarkEvaluationConfig(
            seed=_coerce_required_int(evaluation.get("seed", 0), "evaluation.seed"),
            device=_coerce_string_choice(
                evaluation.get("device", "auto"),
                "evaluation.device",
                _SUPPORTED_BACKEND_BENCHMARK_DEVICES,
            ),
            onnxruntime_provider=_coerce_string_choice(
                evaluation.get("onnxruntime_provider", "auto"),
                "evaluation.onnxruntime_provider",
                _SUPPORTED_ONNXRUNTIME_PROVIDERS,
            ),
            warmup_iterations=_coerce_non_negative_int(
                evaluation.get("warmup_iterations", 10),
                "evaluation.warmup_iterations",
            ),
            benchmark_iterations=_coerce_required_int(
                evaluation.get("benchmark_iterations", 50),
                "evaluation.benchmark_iterations",
            ),
            backends=tuple(
                _coerce_backend_list(
                    evaluation.get("backends", ["torch", "onnxruntime", "tensorrt"])
                )
            ),
            max_mean_abs_diff=_coerce_non_negative_float(
                evaluation.get("max_mean_abs_diff", 0.01),
                "evaluation.max_mean_abs_diff",
            ),
            max_cosine_distance=_coerce_non_negative_float(
                evaluation.get("max_cosine_distance", 0.001),
                "evaluation.max_cosine_distance",
            ),
        ),
        workloads=tuple(workloads),
        validation_commands=tuple(
            _coerce_string_list(payload.get("validation_commands", []), "validation_commands")
        ),
        notes=tuple(_coerce_string_list(payload.get("notes", []), "notes")),
    )


def _coerce_table(raw: object, field_name: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a TOML table.")
    return {str(key): value for key, value in raw.items()}


def _coerce_required_int(raw: object, field_name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{field_name} must be an integer.")
    return raw


def _coerce_non_negative_int(raw: object, field_name: str) -> int:
    value = _coerce_required_int(raw, field_name)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return value


def _coerce_non_negative_float(raw: object, field_name: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{field_name} must be a number.")
    value = float(raw)
    if value < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")
    return value


def _coerce_optional_string(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    return value or None


def _coerce_string_choice(raw: object, field_name: str, choices: frozenset[str]) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{field_name} must be a string.")
    value = raw.strip().lower()
    if value not in choices:
        raise ValueError(f"{field_name} must be one of {sorted(choices)}.")
    return value


def _coerce_string_list(raw: object, field_name: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be an array of strings.")
    values: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{index}] must be a string.")
        stripped = item.strip()
        if not stripped:
            raise ValueError(f"{field_name}[{index}] must be a non-empty string.")
        values.append(stripped)
    return values


def _coerce_backend_list(raw: object) -> list[str]:
    values = _coerce_string_list(raw, "evaluation.backends")
    normalized: list[str] = []
    for value in values:
        backend = value.lower()
        if backend not in _SUPPORTED_BACKEND_BENCHMARK_BACKENDS:
            raise ValueError(
                "evaluation.backends entries must be one of "
                f"{sorted(_SUPPORTED_BACKEND_BENCHMARK_BACKENDS)}."
            )
        normalized.append(backend)
    return normalized


__all__ = [
    "BackendBenchmarkArtifactsConfig",
    "BackendBenchmarkConfig",
    "BackendBenchmarkEvaluationConfig",
    "BackendBenchmarkWorkloadConfig",
    "load_backend_benchmark_config",
]
