"""Build release-oriented inference stress reports and reproducible input assets."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import cycle, islice
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf

from kryptonite.config import ProjectConfig
from kryptonite.deployment import resolve_project_path

from .http import create_http_server
from .stress_memory import (
    StressMemoryMeasurement,
    finish_memory_measurement,
    start_memory_measurement,
)

STRESS_INPUT_CATALOG_NAME = "stress_input_catalog.json"
REPORT_JSON_NAME = "inference_stress_report.json"
REPORT_MARKDOWN_NAME = "inference_stress_report.md"
DEFAULT_STAGE = "demo"
DEFAULT_VERIFY_THRESHOLD = 0.995
DEFAULT_BATCH_SIZES = (1, 4, 8, 16)
DEFAULT_BENCHMARK_ITERATIONS = 2
DEFAULT_WARMUP_ITERATIONS = 1
DEFAULT_SAMPLE_RATE_HZ = 16_000
RUNTIME_ENROLLMENT_ID = "stress_alpha"


@dataclass(frozen=True, slots=True)
class StressInputDescriptor:
    scenario_id: str
    category: str
    audio_path: str
    duration_seconds: float | None
    notes: str

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "audio_path": self.audio_path,
            "duration_seconds": self.duration_seconds,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class GeneratedStressInputs:
    input_root: str
    catalog_path: str
    inputs: tuple[StressInputDescriptor, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "input_root": self.input_root,
            "catalog_path": self.catalog_path,
            "inputs": [descriptor.to_dict() for descriptor in self.inputs],
        }


@dataclass(frozen=True, slots=True)
class StressServiceMetadata:
    service_status: str
    selected_backend: str
    implementation: str
    model_version: str
    platform: str
    python_version: str

    def to_dict(self) -> dict[str, object]:
        return {
            "service_status": self.service_status,
            "selected_backend": self.selected_backend,
            "implementation": self.implementation,
            "model_version": self.model_version,
            "platform": self.platform,
            "python_version": self.python_version,
        }


@dataclass(frozen=True, slots=True)
class AudioStressScenarioResult:
    scenario_id: str
    category: str
    audio_path: str
    notes: str
    status: str
    latency_seconds: float | None
    duration_seconds: float | None
    chunk_count: int | None
    score: float | None
    decision: bool | None
    memory: StressMemoryMeasurement | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "audio_path": self.audio_path,
            "notes": self.notes,
            "status": self.status,
            "latency_seconds": self.latency_seconds,
            "duration_seconds": self.duration_seconds,
            "chunk_count": self.chunk_count,
            "score": self.score,
            "decision": self.decision,
            "memory": (None if self.memory is None else self.memory.to_dict()),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class BatchBurstScenarioResult:
    batch_size: int
    status: str
    iterations: int
    warmup_iterations: int
    mean_iteration_seconds: float | None
    mean_ms_per_audio: float | None
    total_audio_duration_seconds: float | None
    total_chunk_count: int | None
    memory: StressMemoryMeasurement | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "status": self.status,
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "mean_iteration_seconds": self.mean_iteration_seconds,
            "mean_ms_per_audio": self.mean_ms_per_audio,
            "total_audio_duration_seconds": self.total_audio_duration_seconds,
            "total_chunk_count": self.total_chunk_count,
            "memory": (None if self.memory is None else self.memory.to_dict()),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class MalformedRequestResult:
    scenario_id: str
    endpoint: str
    expected_status_code: int
    actual_status_code: int
    matched_expectation: bool
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "endpoint": self.endpoint,
            "expected_status_code": self.expected_status_code,
            "actual_status_code": self.actual_status_code,
            "matched_expectation": self.matched_expectation,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class StressHardLimitSummary:
    validated_stage: str
    verify_threshold: float
    max_full_utterance_seconds: float
    chunk_seconds: float
    chunk_overlap_seconds: float
    min_validated_duration_seconds: float | None
    max_validated_duration_seconds: float | None
    largest_validated_batch_size: int
    largest_validated_total_chunk_count: int
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "validated_stage": self.validated_stage,
            "verify_threshold": self.verify_threshold,
            "max_full_utterance_seconds": self.max_full_utterance_seconds,
            "chunk_seconds": self.chunk_seconds,
            "chunk_overlap_seconds": self.chunk_overlap_seconds,
            "min_validated_duration_seconds": self.min_validated_duration_seconds,
            "max_validated_duration_seconds": self.max_validated_duration_seconds,
            "largest_validated_batch_size": self.largest_validated_batch_size,
            "largest_validated_total_chunk_count": self.largest_validated_total_chunk_count,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class InferenceStressSummary:
    passed: bool
    audio_scenario_count: int
    successful_audio_scenario_count: int
    batch_burst_count: int
    successful_batch_burst_count: int
    malformed_case_count: int
    expected_error_case_count: int
    control_ordering_passed: bool
    long_audio_chunking_observed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "audio_scenario_count": self.audio_scenario_count,
            "successful_audio_scenario_count": self.successful_audio_scenario_count,
            "batch_burst_count": self.batch_burst_count,
            "successful_batch_burst_count": self.successful_batch_burst_count,
            "malformed_case_count": self.malformed_case_count,
            "expected_error_case_count": self.expected_error_case_count,
            "control_ordering_passed": self.control_ordering_passed,
            "long_audio_chunking_observed": self.long_audio_chunking_observed,
        }


@dataclass(frozen=True, slots=True)
class InferenceStressMemorySummary:
    peak_process_rss_mib: float | None
    peak_process_rss_delta_mib: float | None
    peak_cuda_allocated_mib: float | None
    peak_cuda_reserved_mib: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "peak_process_rss_mib": self.peak_process_rss_mib,
            "peak_process_rss_delta_mib": self.peak_process_rss_delta_mib,
            "peak_cuda_allocated_mib": self.peak_cuda_allocated_mib,
            "peak_cuda_reserved_mib": self.peak_cuda_reserved_mib,
        }


@dataclass(frozen=True, slots=True)
class InferenceStressReport:
    inputs: GeneratedStressInputs
    service: StressServiceMetadata
    summary: InferenceStressSummary
    memory: InferenceStressMemorySummary
    audio_scenarios: tuple[AudioStressScenarioResult, ...]
    batch_bursts: tuple[BatchBurstScenarioResult, ...]
    malformed_requests: tuple[MalformedRequestResult, ...]
    hard_limits: StressHardLimitSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "inputs": self.inputs.to_dict(),
            "service": self.service.to_dict(),
            "summary": self.summary.to_dict(),
            "memory": self.memory.to_dict(),
            "audio_scenarios": [result.to_dict() for result in self.audio_scenarios],
            "batch_bursts": [result.to_dict() for result in self.batch_bursts],
            "malformed_requests": [result.to_dict() for result in self.malformed_requests],
            "hard_limits": self.hard_limits.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class WrittenInferenceStressReport:
    output_root: str
    report_json_path: str
    report_markdown_path: str
    summary: InferenceStressSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "summary": self.summary.to_dict(),
        }


def generate_inference_stress_inputs(
    *,
    project_root: Path | str,
    artifacts_root: Path | str = "artifacts",
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
) -> GeneratedStressInputs:
    project_root_path = resolve_project_path(str(project_root), ".")
    input_root = resolve_project_path(
        str(project_root_path),
        str(Path(str(artifacts_root)) / "inference-stress" / "inputs"),
    )
    input_root.mkdir(parents=True, exist_ok=True)

    descriptors = [
        _write_stress_waveform(
            input_root=input_root,
            project_root=project_root_path,
            scenario_id="alpha_enroll_a",
            category="enrollment",
            waveform=_build_tone_waveform(
                frequency_hz=220.0,
                duration_seconds=1.0,
                sample_rate_hz=sample_rate_hz,
            ),
            sample_rate_hz=sample_rate_hz,
            notes="Reference enrollment tone A.",
        ),
        _write_stress_waveform(
            input_root=input_root,
            project_root=project_root_path,
            scenario_id="alpha_enroll_b",
            category="enrollment",
            waveform=_build_tone_waveform(
                frequency_hz=235.0,
                duration_seconds=1.0,
                sample_rate_hz=sample_rate_hz,
            ),
            sample_rate_hz=sample_rate_hz,
            notes="Reference enrollment tone B.",
        ),
        _write_stress_waveform(
            input_root=input_root,
            project_root=project_root_path,
            scenario_id="alpha_reference",
            category="baseline",
            waveform=_build_tone_waveform(
                frequency_hz=250.0,
                duration_seconds=1.0,
                sample_rate_hz=sample_rate_hz,
            ),
            sample_rate_hz=sample_rate_hz,
            notes="Clean same-speaker control for runtime enrollment.",
        ),
        _write_stress_waveform(
            input_root=input_root,
            project_root=project_root_path,
            scenario_id="bravo_reference",
            category="baseline",
            waveform=_build_tone_waveform(
                frequency_hz=360.0,
                duration_seconds=1.0,
                sample_rate_hz=sample_rate_hz,
            ),
            sample_rate_hz=sample_rate_hz,
            notes="Cross-speaker negative control.",
        ),
        _write_stress_waveform(
            input_root=input_root,
            project_root=project_root_path,
            scenario_id="alpha_short",
            category="extreme_duration",
            waveform=_build_tone_waveform(
                frequency_hz=250.0,
                duration_seconds=0.25,
                sample_rate_hz=sample_rate_hz,
            ),
            sample_rate_hz=sample_rate_hz,
            notes="Short 250 ms clip below the nominal 1 s training floor.",
        ),
        _write_stress_waveform(
            input_root=input_root,
            project_root=project_root_path,
            scenario_id="alpha_long_bursty",
            category="extreme_duration",
            waveform=_build_bursty_waveform(
                frequency_hz=250.0,
                repeat_count=12,
                tone_seconds=0.8,
                pause_seconds=0.2,
                sample_rate_hz=sample_rate_hz,
            ),
            sample_rate_hz=sample_rate_hz,
            notes="Long 12 s bursty clip that must trigger multi-window chunking.",
        ),
        _write_stress_waveform(
            input_root=input_root,
            project_root=project_root_path,
            scenario_id="alpha_noisy",
            category="corruption",
            waveform=_build_noisy_waveform(
                frequency_hz=250.0,
                duration_seconds=1.0,
                snr_db=-5.0,
                sample_rate_hz=sample_rate_hz,
                seed=42,
            ),
            sample_rate_hz=sample_rate_hz,
            notes="Same-speaker clip with deterministic white noise at -5 dB SNR.",
        ),
        _write_stress_waveform(
            input_root=input_root,
            project_root=project_root_path,
            scenario_id="alpha_echo",
            category="corruption",
            waveform=_build_echo_waveform(
                frequency_hz=250.0,
                duration_seconds=1.2,
                delay_seconds=0.12,
                decay=0.65,
                sample_rate_hz=sample_rate_hz,
            ),
            sample_rate_hz=sample_rate_hz,
            notes="Same-speaker clip with deterministic synthetic echo tail.",
        ),
        _write_stress_waveform(
            input_root=input_root,
            project_root=project_root_path,
            scenario_id="alpha_clipped",
            category="corruption",
            waveform=_build_clipped_waveform(
                frequency_hz=250.0,
                duration_seconds=1.0,
                drive=2.6,
                sample_rate_hz=sample_rate_hz,
            ),
            sample_rate_hz=sample_rate_hz,
            notes="Same-speaker clip with intentional clipping/saturation.",
        ),
        _write_stress_waveform(
            input_root=input_root,
            project_root=project_root_path,
            scenario_id="alpha_silence",
            category="corruption",
            waveform=np.zeros(sample_rate_hz, dtype=np.float32),
            sample_rate_hz=sample_rate_hz,
            notes="Silence-only waveform to validate degenerate input handling.",
        ),
    ]

    corrupt_audio_path = input_root / "corrupt_audio.wav"
    corrupt_audio_path.write_text("this is not a valid audio container\n", encoding="utf-8")
    descriptors.append(
        StressInputDescriptor(
            scenario_id="corrupt_audio",
            category="malformed",
            audio_path=_relative_to_project(corrupt_audio_path, project_root_path),
            duration_seconds=None,
            notes="Deliberately invalid .wav file used for malformed-input coverage.",
        )
    )
    catalog_path = input_root / STRESS_INPUT_CATALOG_NAME
    catalog_path.write_text(
        json.dumps({"inputs": [descriptor.to_dict() for descriptor in descriptors]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return GeneratedStressInputs(
        input_root=_relative_to_project(input_root, project_root_path),
        catalog_path=_relative_to_project(catalog_path, project_root_path),
        inputs=tuple(descriptors),
    )


def build_inference_stress_report(
    *,
    config: ProjectConfig,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    benchmark_iterations: int = DEFAULT_BENCHMARK_ITERATIONS,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    verify_threshold: float = DEFAULT_VERIFY_THRESHOLD,
    stage: str = DEFAULT_STAGE,
) -> InferenceStressReport:
    if benchmark_iterations <= 0:
        raise ValueError("benchmark_iterations must be positive.")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative.")
    if verify_threshold <= 0.0:
        raise ValueError("verify_threshold must be positive.")

    normalized_batch_sizes = _normalize_batch_sizes(batch_sizes)
    assets = generate_inference_stress_inputs(
        project_root=config.paths.project_root,
        artifacts_root=config.paths.artifacts_root,
    )
    asset_map = {descriptor.scenario_id: descriptor for descriptor in assets.inputs}

    server, thread = _start_server(config=config)
    try:
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        health_payload = _get_json(f"{base_url}/health")
        _enroll_runtime_reference(
            base_url=base_url,
            asset_map=asset_map,
            stage=stage,
        )
        audio_scenarios = tuple(
            _run_audio_scenario(
                base_url=base_url,
                descriptor=asset_map[scenario_id],
                stage=stage,
                verify_threshold=verify_threshold,
            )
            for scenario_id in (
                "alpha_reference",
                "bravo_reference",
                "alpha_short",
                "alpha_long_bursty",
                "alpha_noisy",
                "alpha_echo",
                "alpha_clipped",
                "alpha_silence",
            )
        )
        batch_bursts = tuple(
            _run_batch_burst(
                base_url=base_url,
                batch_size=batch_size,
                stage=stage,
                benchmark_iterations=benchmark_iterations,
                warmup_iterations=warmup_iterations,
                audio_pool=[
                    asset_map["alpha_reference"].audio_path,
                    asset_map["alpha_short"].audio_path,
                    asset_map["alpha_long_bursty"].audio_path,
                    asset_map["alpha_noisy"].audio_path,
                    asset_map["alpha_echo"].audio_path,
                    asset_map["alpha_clipped"].audio_path,
                ],
            )
            for batch_size in normalized_batch_sizes
        )
        malformed_requests = (
            _run_malformed_case(
                base_url=base_url,
                scenario_id="missing_audio_path",
                endpoint="/verify",
                payload={
                    "enrollment_id": RUNTIME_ENROLLMENT_ID,
                    "audio_path": str(Path(assets.input_root) / "missing_audio.wav"),
                    "stage": stage,
                    "threshold": verify_threshold,
                },
                expected_status_code=400,
            ),
            _run_malformed_case(
                base_url=base_url,
                scenario_id="corrupt_audio",
                endpoint="/verify",
                payload={
                    "enrollment_id": RUNTIME_ENROLLMENT_ID,
                    "audio_path": asset_map["corrupt_audio"].audio_path,
                    "stage": stage,
                    "threshold": verify_threshold,
                },
                expected_status_code=400,
            ),
            _run_malformed_case(
                base_url=base_url,
                scenario_id="invalid_stage",
                endpoint="/verify",
                payload={
                    "enrollment_id": RUNTIME_ENROLLMENT_ID,
                    "audio_path": asset_map["alpha_reference"].audio_path,
                    "stage": "release",
                    "threshold": verify_threshold,
                },
                expected_status_code=400,
            ),
            _run_malformed_case(
                base_url=base_url,
                scenario_id="invalid_schema",
                endpoint="/verify",
                payload={"enrollment_id": RUNTIME_ENROLLMENT_ID},
                expected_status_code=422,
            ),
        )
    finally:
        _stop_server(server, thread)

    service = StressServiceMetadata(
        service_status=str(health_payload["status"]),
        selected_backend=str(health_payload["selected_backend"]),
        implementation=str(health_payload["inferencer"]["implementation"]),
        model_version=str(health_payload["model_bundle"]["model_version"]),
        platform=str(health_payload["runtime"]["platform"]),
        python_version=str(health_payload["runtime"]["python_version"]),
    )
    summary = _build_summary(
        audio_scenarios=audio_scenarios,
        batch_bursts=batch_bursts,
        malformed_requests=malformed_requests,
    )
    memory = _build_memory_summary(audio_scenarios=audio_scenarios, batch_bursts=batch_bursts)
    hard_limits = _build_hard_limits(
        config=config,
        audio_scenarios=audio_scenarios,
        batch_bursts=batch_bursts,
        malformed_requests=malformed_requests,
        stage=stage,
        verify_threshold=verify_threshold,
    )
    return InferenceStressReport(
        inputs=assets,
        service=service,
        summary=summary,
        memory=memory,
        audio_scenarios=audio_scenarios,
        batch_bursts=batch_bursts,
        malformed_requests=malformed_requests,
        hard_limits=hard_limits,
    )


def render_inference_stress_markdown(report: InferenceStressReport) -> str:
    lines = [
        "# Inference Stress Report",
        "",
        "## Summary",
        "",
        f"- Status: {'PASS' if report.summary.passed else 'FAIL'}",
        f"- Backend: `{report.service.selected_backend}` / `{report.service.implementation}`",
        f"- Model version: `{report.service.model_version}`",
        f"- Control ordering passed: `{report.summary.control_ordering_passed}`",
        f"- Long-audio chunking observed: `{report.summary.long_audio_chunking_observed}`",
        f"- Peak process RSS: `{_format_float(report.memory.peak_process_rss_mib, digits=3)}` MiB",
        (
            "- Peak process RSS delta: "
            f"`{_format_float(report.memory.peak_process_rss_delta_mib, digits=3)}` MiB"
        ),
        (
            "- Peak CUDA allocated: "
            f"`{_format_float(report.memory.peak_cuda_allocated_mib, digits=3)}` MiB"
        ),
        "",
        "## Audio Stress Scenarios",
        "",
        _markdown_table(
            headers=[
                "Scenario",
                "Category",
                "Status",
                "Duration (s)",
                "Chunks",
                "Score",
                "Decision",
            ],
            rows=[
                [
                    result.scenario_id,
                    result.category,
                    result.status,
                    _format_float(result.duration_seconds, digits=3),
                    _format_int(result.chunk_count),
                    _format_float(result.score, digits=6),
                    _format_bool(result.decision),
                ]
                for result in report.audio_scenarios
            ],
        ),
        "",
        "## Batch Burst Benchmark",
        "",
        _markdown_table(
            headers=[
                "Batch",
                "Status",
                "Mean iter (s)",
                "Mean ms/audio",
                "Chunks",
                "Peak RSS MiB",
                "Peak CUDA MiB",
            ],
            rows=[
                [
                    str(result.batch_size),
                    result.status,
                    _format_float(result.mean_iteration_seconds, digits=6),
                    _format_float(result.mean_ms_per_audio, digits=6),
                    _format_int(result.total_chunk_count),
                    _format_float(
                        None if result.memory is None else result.memory.process_peak_rss_mib,
                        digits=3,
                    ),
                    _format_float(
                        None if result.memory is None else result.memory.cuda_peak_allocated_mib,
                        digits=3,
                    ),
                ]
                for result in report.batch_bursts
            ],
        ),
        "",
        "## Malformed Request Matrix",
        "",
        _markdown_table(
            headers=["Scenario", "Endpoint", "Expected", "Actual", "Matched"],
            rows=[
                [
                    result.scenario_id,
                    result.endpoint,
                    str(result.expected_status_code),
                    str(result.actual_status_code),
                    _format_bool(result.matched_expectation),
                ]
                for result in report.malformed_requests
            ],
        ),
        "",
        "## Hard Limits",
        "",
        f"- Validated stage: `{report.hard_limits.validated_stage}`",
        f"- Verify threshold: `{report.hard_limits.verify_threshold:.6f}`",
        (
            "- Duration envelope exercised: "
            f"`{_format_float(report.hard_limits.min_validated_duration_seconds, digits=3)}` s -> "
            f"`{_format_float(report.hard_limits.max_validated_duration_seconds, digits=3)}` s"
        ),
        (
            "- Demo chunking contract: "
            f"full utterance <= `{report.hard_limits.max_full_utterance_seconds:.3f}` s, "
            f"window `{report.hard_limits.chunk_seconds:.3f}` s, "
            f"overlap `{report.hard_limits.chunk_overlap_seconds:.3f}` s"
        ),
        f"- Largest validated burst: `{report.hard_limits.largest_validated_batch_size}` audios",
        (
            "- Largest validated total chunk count: "
            f"`{report.hard_limits.largest_validated_total_chunk_count}`"
        ),
    ]
    if report.hard_limits.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in report.hard_limits.notes)
    return "\n".join(lines)


def write_inference_stress_report(
    report: InferenceStressReport,
    *,
    output_root: Path | str,
) -> WrittenInferenceStressReport:
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    json_path = output_root_path / REPORT_JSON_NAME
    markdown_path = output_root_path / REPORT_MARKDOWN_NAME
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_inference_stress_markdown(report) + "\n", encoding="utf-8")
    return WrittenInferenceStressReport(
        output_root=str(output_root_path),
        report_json_path=str(json_path),
        report_markdown_path=str(markdown_path),
        summary=report.summary,
    )


def default_stress_report_output_root(*, config: ProjectConfig) -> Path:
    artifacts_root = resolve_project_path(config.paths.project_root, config.paths.artifacts_root)
    return artifacts_root / "inference-stress" / "report"


def _run_audio_scenario(
    *,
    base_url: str,
    descriptor: StressInputDescriptor,
    stage: str,
    verify_threshold: float,
) -> AudioStressScenarioResult:
    payload = {
        "enrollment_id": RUNTIME_ENROLLMENT_ID,
        "audio_path": descriptor.audio_path,
        "stage": stage,
        "threshold": verify_threshold,
    }
    memory_baseline = start_memory_measurement()
    started = time.perf_counter()
    try:
        status_code, response_payload = _post_json_allow_error(f"{base_url}/verify", payload)
    except Exception as exc:  # pragma: no cover - defensive guard around HTTP transport
        return AudioStressScenarioResult(
            scenario_id=descriptor.scenario_id,
            category=descriptor.category,
            audio_path=descriptor.audio_path,
            notes=descriptor.notes,
            status="failed",
            latency_seconds=None,
            duration_seconds=descriptor.duration_seconds,
            chunk_count=None,
            score=None,
            decision=None,
            memory=finish_memory_measurement(memory_baseline),
            error=f"{type(exc).__name__}: {exc}",
        )

    latency_seconds = time.perf_counter() - started
    memory = finish_memory_measurement(memory_baseline)
    if status_code != 200:
        return AudioStressScenarioResult(
            scenario_id=descriptor.scenario_id,
            category=descriptor.category,
            audio_path=descriptor.audio_path,
            notes=descriptor.notes,
            status="failed",
            latency_seconds=round(latency_seconds, 8),
            duration_seconds=descriptor.duration_seconds,
            chunk_count=None,
            score=None,
            decision=None,
            memory=memory,
            error=_extract_message(response_payload),
        )

    probe_item = response_payload["probe_items"][0]
    return AudioStressScenarioResult(
        scenario_id=descriptor.scenario_id,
        category=descriptor.category,
        audio_path=descriptor.audio_path,
        notes=descriptor.notes,
        status="passed",
        latency_seconds=round(latency_seconds, 8),
        duration_seconds=float(probe_item["duration_seconds"]),
        chunk_count=int(probe_item["chunk_count"]),
        score=float(response_payload["scores"][0]),
        decision=bool(response_payload["decisions"][0]),
        memory=memory,
        error=None,
    )


def _run_batch_burst(
    *,
    base_url: str,
    batch_size: int,
    stage: str,
    benchmark_iterations: int,
    warmup_iterations: int,
    audio_pool: Sequence[str],
) -> BatchBurstScenarioResult:
    audio_paths = list(islice(cycle(audio_pool), batch_size))
    memory_baseline = start_memory_measurement()
    status_code, payload = _post_json_allow_error(
        f"{base_url}/benchmark",
        {
            "audio_paths": audio_paths,
            "stage": stage,
            "iterations": benchmark_iterations,
            "warmup_iterations": warmup_iterations,
        },
    )
    memory = finish_memory_measurement(memory_baseline)
    if status_code != 200:
        return BatchBurstScenarioResult(
            batch_size=batch_size,
            status="failed",
            iterations=benchmark_iterations,
            warmup_iterations=warmup_iterations,
            mean_iteration_seconds=None,
            mean_ms_per_audio=None,
            total_audio_duration_seconds=None,
            total_chunk_count=None,
            memory=memory,
            error=_extract_message(payload),
        )
    return BatchBurstScenarioResult(
        batch_size=batch_size,
        status="passed",
        iterations=int(payload["iterations"]),
        warmup_iterations=int(payload["warmup_iterations"]),
        mean_iteration_seconds=float(payload["mean_iteration_seconds"]),
        mean_ms_per_audio=float(payload["mean_ms_per_audio"]),
        total_audio_duration_seconds=float(payload["total_audio_duration_seconds"]),
        total_chunk_count=int(payload["total_chunk_count"]),
        memory=memory,
        error=None,
    )


def _run_malformed_case(
    *,
    base_url: str,
    scenario_id: str,
    endpoint: str,
    payload: dict[str, object],
    expected_status_code: int,
) -> MalformedRequestResult:
    actual_status_code, response_payload = _post_json_allow_error(f"{base_url}{endpoint}", payload)
    return MalformedRequestResult(
        scenario_id=scenario_id,
        endpoint=endpoint,
        expected_status_code=expected_status_code,
        actual_status_code=actual_status_code,
        matched_expectation=actual_status_code == expected_status_code,
        message=_extract_message(response_payload),
    )


def _build_summary(
    *,
    audio_scenarios: Sequence[AudioStressScenarioResult],
    batch_bursts: Sequence[BatchBurstScenarioResult],
    malformed_requests: Sequence[MalformedRequestResult],
) -> InferenceStressSummary:
    successful_audio = [result for result in audio_scenarios if result.status == "passed"]
    successful_batches = [result for result in batch_bursts if result.status == "passed"]
    expected_errors = [result for result in malformed_requests if result.matched_expectation]
    scenario_map = {result.scenario_id: result for result in audio_scenarios}
    control_ordering_passed = False
    if (
        scenario_map.get("alpha_reference") is not None
        and scenario_map.get("bravo_reference") is not None
        and scenario_map["alpha_reference"].score is not None
        and scenario_map["bravo_reference"].score is not None
    ):
        control_ordering_passed = (
            scenario_map["alpha_reference"].score > scenario_map["bravo_reference"].score
        )
    long_audio_chunking_observed = (
        scenario_map.get("alpha_long_bursty") is not None
        and (scenario_map["alpha_long_bursty"].chunk_count or 0) > 1
    )
    passed = (
        len(successful_audio) == len(audio_scenarios)
        and len(successful_batches) == len(batch_bursts)
        and len(expected_errors) == len(malformed_requests)
        and control_ordering_passed
        and long_audio_chunking_observed
    )
    return InferenceStressSummary(
        passed=passed,
        audio_scenario_count=len(audio_scenarios),
        successful_audio_scenario_count=len(successful_audio),
        batch_burst_count=len(batch_bursts),
        successful_batch_burst_count=len(successful_batches),
        malformed_case_count=len(malformed_requests),
        expected_error_case_count=len(expected_errors),
        control_ordering_passed=control_ordering_passed,
        long_audio_chunking_observed=long_audio_chunking_observed,
    )


def _build_memory_summary(
    *,
    audio_scenarios: Sequence[AudioStressScenarioResult],
    batch_bursts: Sequence[BatchBurstScenarioResult],
) -> InferenceStressMemorySummary:
    measurements = [
        result.memory for result in [*audio_scenarios, *batch_bursts] if result.memory is not None
    ]
    return InferenceStressMemorySummary(
        peak_process_rss_mib=_max_memory_value(
            measurement.process_peak_rss_mib for measurement in measurements
        ),
        peak_process_rss_delta_mib=_max_memory_value(
            measurement.process_peak_rss_delta_mib for measurement in measurements
        ),
        peak_cuda_allocated_mib=_max_memory_value(
            measurement.cuda_peak_allocated_mib for measurement in measurements
        ),
        peak_cuda_reserved_mib=_max_memory_value(
            measurement.cuda_peak_reserved_mib for measurement in measurements
        ),
    )


def _build_hard_limits(
    *,
    config: ProjectConfig,
    audio_scenarios: Sequence[AudioStressScenarioResult],
    batch_bursts: Sequence[BatchBurstScenarioResult],
    malformed_requests: Sequence[MalformedRequestResult],
    stage: str,
    verify_threshold: float,
) -> StressHardLimitSummary:
    successful_audio = [result for result in audio_scenarios if result.status == "passed"]
    successful_batches = [result for result in batch_bursts if result.status == "passed"]
    durations = [
        result.duration_seconds
        for result in successful_audio
        if result.duration_seconds is not None
    ]
    chunk_totals = [
        result.total_chunk_count
        for result in successful_batches
        if result.total_chunk_count is not None
    ]
    notes = [
        (
            "Synthetic stress inputs are deterministic and live under "
            "artifacts/inference-stress/inputs."
        ),
        (
            "Malformed-input expectations are part of the contract: missing/corrupt "
            "audio should fail with 400 and schema errors with 422."
        ),
        (
            "This report validates serving-path hard limits and runtime behavior; "
            "it does not replace offline corrupted-dev quality evaluation."
        ),
    ]
    if any(not result.matched_expectation for result in malformed_requests):
        notes.append("At least one malformed-input case returned an unexpected status code.")
    return StressHardLimitSummary(
        validated_stage=stage,
        verify_threshold=verify_threshold,
        max_full_utterance_seconds=float(config.chunking.demo_max_full_utterance_seconds),
        chunk_seconds=float(config.chunking.demo_chunk_seconds),
        chunk_overlap_seconds=float(config.chunking.demo_chunk_overlap_seconds),
        min_validated_duration_seconds=min(durations) if durations else None,
        max_validated_duration_seconds=max(durations) if durations else None,
        largest_validated_batch_size=max(
            (result.batch_size for result in successful_batches), default=0
        ),
        largest_validated_total_chunk_count=max(chunk_totals, default=0),
        notes=tuple(notes),
    )


def _enroll_runtime_reference(
    *,
    base_url: str,
    asset_map: dict[str, StressInputDescriptor],
    stage: str,
) -> None:
    status_code, payload = _post_json_allow_error(
        f"{base_url}/enroll",
        {
            "enrollment_id": RUNTIME_ENROLLMENT_ID,
            "audio_paths": [
                asset_map["alpha_enroll_a"].audio_path,
                asset_map["alpha_enroll_b"].audio_path,
            ],
            "stage": stage,
            "metadata": {"source": "stress_report"},
        },
    )
    if status_code not in {200, 201}:
        raise RuntimeError(
            "Failed to create runtime enrollment for inference stress report: "
            f"{_extract_message(payload)}"
        )


def _normalize_batch_sizes(batch_sizes: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for batch_size in batch_sizes:
        value = int(batch_size)
        if value <= 0:
            raise ValueError("batch_sizes must contain only positive integers.")
        normalized.append(value)
    if not normalized:
        raise ValueError("batch_sizes must not be empty.")
    return tuple(normalized)


def _build_tone_waveform(
    *,
    frequency_hz: float,
    duration_seconds: float,
    sample_rate_hz: int,
    amplitude: float = 0.35,
) -> np.ndarray:
    frame_count = max(1, int(round(duration_seconds * sample_rate_hz)))
    timeline = np.arange(frame_count, dtype=np.float32) / float(sample_rate_hz)
    return (amplitude * np.sin(2.0 * np.pi * frequency_hz * timeline)).astype(np.float32)


def _build_bursty_waveform(
    *,
    frequency_hz: float,
    repeat_count: int,
    tone_seconds: float,
    pause_seconds: float,
    sample_rate_hz: int,
) -> np.ndarray:
    tone = _build_tone_waveform(
        frequency_hz=frequency_hz,
        duration_seconds=tone_seconds,
        sample_rate_hz=sample_rate_hz,
    )
    pause = np.zeros(max(1, int(round(pause_seconds * sample_rate_hz))), dtype=np.float32)
    return np.concatenate([segment for _ in range(repeat_count) for segment in (tone, pause)])


def _build_noisy_waveform(
    *,
    frequency_hz: float,
    duration_seconds: float,
    snr_db: float,
    sample_rate_hz: int,
    seed: int,
) -> np.ndarray:
    clean = _build_tone_waveform(
        frequency_hz=frequency_hz,
        duration_seconds=duration_seconds,
        sample_rate_hz=sample_rate_hz,
    )
    signal_rms = float(np.sqrt(np.mean(np.square(clean))))
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, clean.shape).astype(np.float32)
    noise = noise / max(float(np.sqrt(np.mean(np.square(noise)))), 1e-8)
    noisy = clean + noise * target_noise_rms
    return np.clip(noisy, -1.0, 1.0).astype(np.float32)


def _build_echo_waveform(
    *,
    frequency_hz: float,
    duration_seconds: float,
    delay_seconds: float,
    decay: float,
    sample_rate_hz: int,
) -> np.ndarray:
    clean = _build_tone_waveform(
        frequency_hz=frequency_hz,
        duration_seconds=duration_seconds,
        sample_rate_hz=sample_rate_hz,
    )
    delay_samples = max(1, int(round(delay_seconds * sample_rate_hz)))
    echoed = np.zeros(clean.shape[0] + delay_samples, dtype=np.float32)
    echoed[: clean.shape[0]] += clean
    echoed[delay_samples:] += clean * decay
    peak = max(float(np.abs(echoed).max()), 1.0)
    return (echoed / peak).astype(np.float32) * 0.9


def _build_clipped_waveform(
    *,
    frequency_hz: float,
    duration_seconds: float,
    drive: float,
    sample_rate_hz: int,
) -> np.ndarray:
    clean = _build_tone_waveform(
        frequency_hz=frequency_hz,
        duration_seconds=duration_seconds,
        sample_rate_hz=sample_rate_hz,
        amplitude=0.55,
    )
    return np.clip(clean * drive, -1.0, 1.0).astype(np.float32)


def _write_stress_waveform(
    *,
    input_root: Path,
    project_root: Path,
    scenario_id: str,
    category: str,
    waveform: np.ndarray,
    sample_rate_hz: int,
    notes: str,
) -> StressInputDescriptor:
    path = input_root / f"{scenario_id}.wav"
    sf.write(path, waveform, sample_rate_hz, format="WAV")
    return StressInputDescriptor(
        scenario_id=scenario_id,
        category=category,
        audio_path=_relative_to_project(path, project_root),
        duration_seconds=round(float(waveform.shape[0]) / float(sample_rate_hz), 6),
        notes=notes,
    )


def _start_server(*, config: ProjectConfig) -> tuple[Any, threading.Thread]:
    server = create_http_server(host="127.0.0.1", port=0, config=config)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.wait_started()
    return server, thread


def _stop_server(server: Any, thread: threading.Thread) -> None:
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


def _get_json(url: str) -> dict[str, Any]:
    with urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json_allow_error(url: str, payload: Mapping[str, object]) -> tuple[int, dict[str, Any]]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"status": "error", "message": body or str(exc)}
        return exc.code, payload


def _extract_message(payload: dict[str, Any]) -> str:
    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return message
    return json.dumps(payload, sort_keys=True)


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _markdown_table(*, headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    rendered_rows = list(rows)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rendered_rows)
    return "\n".join(lines)


def _format_float(value: float | None, *, digits: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _format_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def _format_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return "true" if value else "false"


def _max_memory_value(values: Iterable[float | None]) -> float | None:
    resolved = [value for value in values if value is not None]
    if not resolved:
        return None
    return round(max(resolved), 3)


__all__ = [
    "DEFAULT_BATCH_SIZES",
    "DEFAULT_BENCHMARK_ITERATIONS",
    "DEFAULT_VERIFY_THRESHOLD",
    "DEFAULT_WARMUP_ITERATIONS",
    "AudioStressScenarioResult",
    "BatchBurstScenarioResult",
    "GeneratedStressInputs",
    "InferenceStressReport",
    "InferenceStressMemorySummary",
    "InferenceStressSummary",
    "MalformedRequestResult",
    "REPORT_JSON_NAME",
    "REPORT_MARKDOWN_NAME",
    "StressHardLimitSummary",
    "StressInputDescriptor",
    "StressMemoryMeasurement",
    "StressServiceMetadata",
    "WrittenInferenceStressReport",
    "build_inference_stress_report",
    "default_stress_report_output_root",
    "generate_inference_stress_inputs",
    "render_inference_stress_markdown",
    "write_inference_stress_report",
]
