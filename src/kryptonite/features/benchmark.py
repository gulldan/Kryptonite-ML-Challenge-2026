"""Benchmark and policy reporting for Fbank feature caching."""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from kryptonite.config import FeatureCacheConfig, FeaturesConfig, NormalizationConfig, VADConfig
from kryptonite.data import AudioLoadRequest, iter_manifest_audio
from kryptonite.data.audio_loader import LoadedManifestAudio
from kryptonite.deployment import resolve_project_path

from .cache import (
    FeatureCacheMaterializationReport,
    FeatureCacheSettings,
    FeatureCacheStore,
    materialize_feature_cache,
    resolve_feature_cache_root,
)
from .fbank import FbankExtractionRequest, FbankExtractor

REPORT_JSON_NAME = "feature_cache_report.json"
REPORT_MARKDOWN_NAME = "feature_cache_report.md"
REPORT_ROWS_NAME = "feature_cache_rows.jsonl"


@dataclass(frozen=True, slots=True)
class FeatureBenchmarkScenario:
    name: str
    device: str
    benchmark_iterations: int
    warmup_iterations: int
    utterance_count: int
    total_audio_duration_seconds: float
    total_wall_seconds: float
    mean_ms_per_utterance: float
    median_ms_per_utterance: float
    p95_ms_per_utterance: float
    utterances_per_second: float
    audio_seconds_per_second: float
    real_time_factor: float

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "device": self.device,
            "benchmark_iterations": self.benchmark_iterations,
            "warmup_iterations": self.warmup_iterations,
            "utterance_count": self.utterance_count,
            "total_audio_duration_seconds": self.total_audio_duration_seconds,
            "total_wall_seconds": self.total_wall_seconds,
            "mean_ms_per_utterance": self.mean_ms_per_utterance,
            "median_ms_per_utterance": self.median_ms_per_utterance,
            "p95_ms_per_utterance": self.p95_ms_per_utterance,
            "utterances_per_second": self.utterances_per_second,
            "audio_seconds_per_second": self.audio_seconds_per_second,
            "real_time_factor": self.real_time_factor,
        }


@dataclass(frozen=True, slots=True)
class FeatureFrontendPolicyDecision:
    train_policy: str
    train_rationale: tuple[str, ...]
    dev_policy: str
    dev_rationale: tuple[str, ...]
    infer_policy: str
    infer_rationale: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "train_policy": self.train_policy,
            "train_rationale": list(self.train_rationale),
            "dev_policy": self.dev_policy,
            "dev_rationale": list(self.dev_rationale),
            "infer_policy": self.infer_policy,
            "infer_rationale": list(self.infer_rationale),
        }


@dataclass(frozen=True, slots=True)
class FeatureCacheBenchmarkReport:
    project_root: str
    manifest_path: str
    selected_device: str
    limit: int | None
    request: FbankExtractionRequest
    settings: FeatureCacheSettings
    materialization: FeatureCacheMaterializationReport
    benchmarks: list[FeatureBenchmarkScenario]
    policy: FeatureFrontendPolicyDecision

    def to_dict(self, *, include_records: bool = False) -> dict[str, object]:
        return {
            "project_root": self.project_root,
            "manifest_path": self.manifest_path,
            "selected_device": self.selected_device,
            "limit": self.limit,
            "request": {
                "sample_rate_hz": self.request.sample_rate_hz,
                "num_mel_bins": self.request.num_mel_bins,
                "frame_length_ms": self.request.frame_length_ms,
                "frame_shift_ms": self.request.frame_shift_ms,
                "fft_size": self.request.fft_size,
                "window_type": self.request.window_type,
                "f_min_hz": self.request.f_min_hz,
                "f_max_hz": self.request.f_max_hz,
                "power": self.request.power,
                "log_offset": self.request.log_offset,
                "pad_end": self.request.pad_end,
                "cmvn_mode": self.request.cmvn_mode,
                "cmvn_window_frames": self.request.cmvn_window_frames,
                "output_dtype": self.request.output_dtype,
            },
            "settings": {
                "namespace": self.settings.namespace,
                "train_policy": self.settings.train_policy,
                "dev_policy": self.settings.dev_policy,
                "infer_policy": self.settings.infer_policy,
                "benchmark_device": self.settings.benchmark_device,
                "benchmark_warmup_iterations": self.settings.benchmark_warmup_iterations,
                "benchmark_iterations": self.settings.benchmark_iterations,
            },
            "materialization": self.materialization.to_dict(include_records=include_records),
            "benchmarks": [scenario.to_dict() for scenario in self.benchmarks],
            "policy": self.policy.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class WrittenFeatureCacheBenchmarkReport:
    output_root: str
    json_path: str
    markdown_path: str
    rows_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_root": self.output_root,
            "json_path": self.json_path,
            "markdown_path": self.markdown_path,
            "rows_path": self.rows_path,
        }


def build_feature_cache_benchmark_report(
    *,
    project_root: Path | str,
    cache_root: Path | str,
    manifest_path: Path | str,
    normalization: NormalizationConfig,
    features: FeaturesConfig,
    feature_cache: FeatureCacheConfig,
    vad: VADConfig | None = None,
    limit: int | None = None,
    benchmark_device: str | None = None,
    force: bool = False,
) -> FeatureCacheBenchmarkReport:
    settings = FeatureCacheSettings.from_config(feature_cache)
    device_preference = (
        settings.normalized_benchmark_device
        if benchmark_device is None
        else benchmark_device.lower()
    )
    selected_device = _resolve_benchmark_device(device_preference)
    project_root_path = resolve_project_path(str(project_root), ".")
    feature_request = FbankExtractionRequest.from_config(features)
    audio_request = AudioLoadRequest.from_config(normalization, vad=vad)
    samples = list(
        iter_manifest_audio(
            manifest_path,
            project_root=project_root_path,
            request=audio_request,
        )
    )
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError("No manifest audio rows were available for feature cache benchmarking")

    store = FeatureCacheStore(
        root=resolve_feature_cache_root(project_root=project_root_path, cache_root=cache_root),
        settings=settings,
    )
    materialization = materialize_feature_cache(
        samples,
        store=store,
        request=feature_request,
        force=force,
    )

    cpu_extractor = FbankExtractor(request=feature_request)
    with tempfile.TemporaryDirectory() as temporary_cache_root:
        benchmark_store = FeatureCacheStore(root=temporary_cache_root, settings=settings)
        benchmark_extractor = FbankExtractor(request=feature_request)
        benchmarks = [
            _benchmark_scenario(
                name="cpu_precompute_write",
                device="cpu",
                samples=samples,
                benchmark_iterations=settings.benchmark_iterations,
                warmup_iterations=settings.benchmark_warmup_iterations,
                operation=lambda sample: _write_features_to_store(
                    store=benchmark_store,
                    extractor=benchmark_extractor,
                    request=feature_request,
                    sample=sample,
                ),
            ),
            _benchmark_scenario(
                name="cpu_runtime_extract",
                device="cpu",
                samples=samples,
                benchmark_iterations=settings.benchmark_iterations,
                warmup_iterations=settings.benchmark_warmup_iterations,
                operation=lambda sample: cpu_extractor.extract(
                    sample.audio.waveform,
                    sample_rate_hz=sample.audio.sample_rate_hz,
                ),
            ),
            _benchmark_scenario(
                name="cpu_cache_read",
                device="cpu",
                samples=samples,
                benchmark_iterations=settings.benchmark_iterations,
                warmup_iterations=settings.benchmark_warmup_iterations,
                operation=lambda sample: store.load(
                    store.build_key(loaded_audio=sample.audio, request=feature_request),
                    device="cpu",
                ),
            ),
        ]

        if selected_device == "cuda":
            gpu_extractor = FbankExtractor(request=feature_request)
            benchmarks.extend(
                [
                    _benchmark_scenario(
                        name="cuda_runtime_extract",
                        device="cuda",
                        samples=samples,
                        benchmark_iterations=settings.benchmark_iterations,
                        warmup_iterations=settings.benchmark_warmup_iterations,
                        operation=lambda sample: gpu_extractor.extract(
                            sample.audio.waveform.to(device="cuda"),
                            sample_rate_hz=sample.audio.sample_rate_hz,
                        ),
                    ),
                    _benchmark_scenario(
                        name="cuda_cache_read",
                        device="cuda",
                        samples=samples,
                        benchmark_iterations=settings.benchmark_iterations,
                        warmup_iterations=settings.benchmark_warmup_iterations,
                        operation=lambda sample: store.load(
                            store.build_key(loaded_audio=sample.audio, request=feature_request),
                            device="cuda",
                        ),
                    ),
                ]
            )

    policy = _build_policy_decision(
        settings=settings,
        selected_device=selected_device,
        benchmarks=benchmarks,
    )
    return FeatureCacheBenchmarkReport(
        project_root=str(project_root_path),
        manifest_path=str(manifest_path),
        selected_device=selected_device,
        limit=limit,
        request=feature_request,
        settings=settings,
        materialization=materialization,
        benchmarks=benchmarks,
        policy=policy,
    )


def write_feature_cache_benchmark_report(
    *,
    report: FeatureCacheBenchmarkReport,
    output_root: Path | str,
) -> WrittenFeatureCacheBenchmarkReport:
    output_root_path = resolve_project_path(report.project_root, str(output_root))
    output_root_path.mkdir(parents=True, exist_ok=True)
    json_path = output_root_path / REPORT_JSON_NAME
    markdown_path = output_root_path / REPORT_MARKDOWN_NAME
    rows_path = output_root_path / REPORT_ROWS_NAME

    json_path.write_text(json.dumps(report.to_dict(include_records=True), indent=2, sort_keys=True))
    markdown_path.write_text(render_feature_cache_benchmark_markdown(report))
    rows_path.write_text(
        "".join(
            json.dumps(record.to_dict(), sort_keys=True) + "\n"
            for record in report.materialization.records
        )
    )
    return WrittenFeatureCacheBenchmarkReport(
        output_root=str(output_root_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
        rows_path=str(rows_path),
    )


def render_feature_cache_benchmark_markdown(report: FeatureCacheBenchmarkReport) -> str:
    benchmark_lines = [
        "| Scenario | Device | Mean ms/utt | P95 ms/utt | Utts/s | Audio s/s | RTF |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in report.benchmarks:
        benchmark_lines.append(
            "| "
            f"{scenario.name} | "
            f"{scenario.device} | "
            f"{scenario.mean_ms_per_utterance:.3f} | "
            f"{scenario.p95_ms_per_utterance:.3f} | "
            f"{scenario.utterances_per_second:.3f} | "
            f"{scenario.audio_seconds_per_second:.3f} | "
            f"{scenario.real_time_factor:.3f} |"
        )

    lines = [
        "# Feature Cache Report",
        "",
        "## Scope",
        "",
        f"- manifest: `{report.manifest_path}`",
        f"- selected benchmark device: `{report.selected_device}`",
        f"- cache root: `{report.materialization.cache_root}`",
        f"- cache namespace: `{report.settings.namespace}`",
        f"- benchmark iterations: `{report.settings.benchmark_iterations}`",
        f"- warmup iterations: `{report.settings.benchmark_warmup_iterations}`",
        "",
        "## Cache Materialization",
        "",
        f"- rows: `{report.materialization.summary.row_count}`",
        f"- cache hits: `{report.materialization.summary.cache_hit_count}`",
        f"- cache writes: `{report.materialization.summary.cache_write_count}`",
        f"- unique cache entries: `{report.materialization.summary.unique_cache_entry_count}`",
        f"- total input duration seconds: "
        f"`{report.materialization.summary.total_input_duration_seconds}`",
        f"- total feature frames: `{report.materialization.summary.total_feature_frames}`",
        f"- total unique cache size bytes: "
        f"`{report.materialization.summary.total_unique_cache_size_bytes}`",
        "",
        "## Benchmarks",
        "",
        *benchmark_lines,
        "",
        "## Policy",
        "",
        f"- train policy: `{report.policy.train_policy}`",
        *[f"  - {line}" for line in report.policy.train_rationale],
        f"- dev policy: `{report.policy.dev_policy}`",
        *[f"  - {line}" for line in report.policy.dev_rationale],
        f"- infer policy: `{report.policy.infer_policy}`",
        *[f"  - {line}" for line in report.policy.infer_rationale],
        "",
        "## Cache Invalidation",
        "",
        "- cache keys include source file size and mtime plus the loader-time audio window,",
        "  VAD/loudness decisions, and the full Fbank request payload",
        "- changing the source audio file or any loader/frontend knob produces a new key",
        "- cache artifacts remain content-addressed under `artifacts/cache/features/<namespace>/`",
    ]
    return "\n".join(lines)


def _build_policy_decision(
    *,
    settings: FeatureCacheSettings,
    selected_device: str,
    benchmarks: list[FeatureBenchmarkScenario],
) -> FeatureFrontendPolicyDecision:
    by_name = {scenario.name: scenario for scenario in benchmarks}
    train_runtime = by_name["cpu_runtime_extract"]
    train_cache = by_name["cpu_cache_read"]

    train_rationale = (
        "Training should precompute features on CPU once and reuse the disk cache across epochs.",
        "This benchmark measured CPU runtime extraction at "
        f"{train_runtime.mean_ms_per_utterance:.3f} ms/utt and cache reads at "
        f"{train_cache.mean_ms_per_utterance:.3f} ms/utt; the policy decision is therefore "
        "about amortizing repeated epochs and freezing the frontend contract, not about a "
        "guaranteed single-pass speedup.",
        "Cache keys include source-file metadata plus loader/frontend fingerprints, so changes "
        "invalidate automatically instead of reusing stale tensors.",
    )
    dev_rationale = (
        "Dev and ablation loops often reread the same held-out utterances, so cache reuse "
        "pays off after the first pass.",
        "The runtime path remains available for one-shot smoke checks when cache prep would "
        "be unnecessary overhead.",
    )

    infer_rationale_lines = [
        "Inference keeps runtime feature extraction as the default because uploaded or "
        "request-scoped audio usually has low cache locality.",
    ]
    if selected_device == "cuda":
        infer_runtime = by_name["cuda_runtime_extract"]
        infer_cache = by_name["cuda_cache_read"]
        infer_rationale_lines.append(
            "On CUDA, runtime extraction measured "
            f"{infer_runtime.mean_ms_per_utterance:.3f} ms/utt versus "
            f"{infer_cache.mean_ms_per_utterance:.3f} ms/utt for disk cache read plus transfer."
        )
    else:
        infer_runtime = by_name["cpu_runtime_extract"]
        infer_rationale_lines.append(
            "The selected benchmark device is CPU, so cache remains an optional dev aid rather "
            "than a serving default."
        )

    return FeatureFrontendPolicyDecision(
        train_policy=settings.train_policy,
        train_rationale=train_rationale,
        dev_policy=settings.dev_policy,
        dev_rationale=dev_rationale,
        infer_policy=settings.infer_policy,
        infer_rationale=tuple(infer_rationale_lines),
    )


def _benchmark_scenario(
    *,
    name: str,
    device: str,
    samples: list[LoadedManifestAudio],
    benchmark_iterations: int,
    warmup_iterations: int,
    operation: Callable[[LoadedManifestAudio], torch.Tensor],
) -> FeatureBenchmarkScenario:
    total_audio_duration = sum(sample.audio.duration_seconds for sample in samples)
    timed_samples_ms: list[float] = []
    total_wall_seconds = 0.0

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            for sample in samples:
                result = operation(sample)
                del result
            _synchronize(device)

        for _ in range(benchmark_iterations):
            for sample in samples:
                _synchronize(device)
                started_at = time.perf_counter()
                result = operation(sample)
                _synchronize(device)
                elapsed_seconds = time.perf_counter() - started_at
                total_wall_seconds += elapsed_seconds
                timed_samples_ms.append(elapsed_seconds * 1000.0)
                del result

    utterance_count = len(samples) * benchmark_iterations
    audio_duration_total = total_audio_duration * benchmark_iterations
    return FeatureBenchmarkScenario(
        name=name,
        device=device,
        benchmark_iterations=benchmark_iterations,
        warmup_iterations=warmup_iterations,
        utterance_count=utterance_count,
        total_audio_duration_seconds=round(audio_duration_total, 6),
        total_wall_seconds=round(total_wall_seconds, 6),
        mean_ms_per_utterance=round(statistics.fmean(timed_samples_ms), 6),
        median_ms_per_utterance=round(statistics.median(timed_samples_ms), 6),
        p95_ms_per_utterance=round(_percentile(timed_samples_ms, 0.95), 6),
        utterances_per_second=round(utterance_count / total_wall_seconds, 6),
        audio_seconds_per_second=round(audio_duration_total / total_wall_seconds, 6),
        real_time_factor=round(total_wall_seconds / audio_duration_total, 6),
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _resolve_benchmark_device(preference: str) -> str:
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preference == "cuda" and not torch.cuda.is_available():
        raise ValueError("benchmark_device='cuda' requested, but CUDA is not available")
    return preference


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _write_features_to_store(
    *,
    store: FeatureCacheStore,
    extractor: FbankExtractor,
    request: FbankExtractionRequest,
    sample: LoadedManifestAudio,
) -> torch.Tensor:
    key = store.build_key(loaded_audio=sample.audio, request=request)
    features = extractor.extract(sample.audio.waveform, sample_rate_hz=sample.audio.sample_rate_hz)
    store.write(
        key,
        features=features,
        loaded_audio=sample.audio,
        request=request,
    )
    return features


__all__ = [
    "FeatureBenchmarkScenario",
    "FeatureCacheBenchmarkReport",
    "FeatureFrontendPolicyDecision",
    "WrittenFeatureCacheBenchmarkReport",
    "build_feature_cache_benchmark_report",
    "render_feature_cache_benchmark_markdown",
    "write_feature_cache_benchmark_report",
]
