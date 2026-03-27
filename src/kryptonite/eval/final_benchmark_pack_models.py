"""Datamodels for the final benchmark pack."""

from __future__ import annotations

from dataclasses import dataclass

FINAL_BENCHMARK_PACK_JSON_NAME = "final_benchmark_pack.json"
FINAL_BENCHMARK_PACK_MARKDOWN_NAME = "final_benchmark_pack.md"
FINAL_BENCHMARK_PACK_CANDIDATES_JSONL_NAME = "final_benchmark_pack_candidates.jsonl"
FINAL_BENCHMARK_PACK_PAIRWISE_JSONL_NAME = "final_benchmark_pack_pairwise.jsonl"


@dataclass(frozen=True, slots=True)
class BenchmarkPackArtifactRef:
    kind: str
    original_path: str
    copied_path: str
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "original_path": self.original_path,
            "copied_path": self.copied_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class CandidateQualitySummary:
    trial_count: int
    positive_count: int
    negative_count: int
    eer: float
    eer_threshold: float
    min_dcf: float
    min_dcf_threshold: float
    mean_positive_score: float | None
    mean_negative_score: float | None
    score_gap: float | None
    balanced_threshold: float | None
    demo_threshold: float | None
    production_threshold: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "trial_count": self.trial_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "eer": self.eer,
            "eer_threshold": self.eer_threshold,
            "min_dcf": self.min_dcf,
            "min_dcf_threshold": self.min_dcf_threshold,
            "mean_positive_score": self.mean_positive_score,
            "mean_negative_score": self.mean_negative_score,
            "score_gap": self.score_gap,
            "balanced_threshold": self.balanced_threshold,
            "demo_threshold": self.demo_threshold,
            "production_threshold": self.production_threshold,
        }


@dataclass(frozen=True, slots=True)
class CandidateOperationalSummary:
    stress_passed: bool
    validated_stage: str
    largest_validated_batch_size: int
    mean_ms_per_audio_at_largest_batch: float | None
    max_validated_duration_seconds: float | None
    largest_validated_total_chunk_count: int
    peak_process_rss_mib: float | None
    peak_process_rss_delta_mib: float | None
    peak_cuda_allocated_mib: float | None
    peak_cuda_reserved_mib: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "stress_passed": self.stress_passed,
            "validated_stage": self.validated_stage,
            "largest_validated_batch_size": self.largest_validated_batch_size,
            "mean_ms_per_audio_at_largest_batch": self.mean_ms_per_audio_at_largest_batch,
            "max_validated_duration_seconds": self.max_validated_duration_seconds,
            "largest_validated_total_chunk_count": self.largest_validated_total_chunk_count,
            "peak_process_rss_mib": self.peak_process_rss_mib,
            "peak_process_rss_delta_mib": self.peak_process_rss_delta_mib,
            "peak_cuda_allocated_mib": self.peak_cuda_allocated_mib,
            "peak_cuda_reserved_mib": self.peak_cuda_reserved_mib,
        }


@dataclass(frozen=True, slots=True)
class CandidateBundleSummary:
    model_version: str
    model_file: str | None
    input_name: str | None
    output_name: str | None
    sample_rate_hz: int | None
    enrollment_cache_compatibility_id: str | None
    export_boundary: str | None
    export_profile: str | None
    frontend_location: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "model_version": self.model_version,
            "model_file": self.model_file,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "sample_rate_hz": self.sample_rate_hz,
            "enrollment_cache_compatibility_id": self.enrollment_cache_compatibility_id,
            "export_boundary": self.export_boundary,
            "export_profile": self.export_profile,
            "frontend_location": self.frontend_location,
        }


@dataclass(frozen=True, slots=True)
class CandidateBenchmarkSummary:
    candidate_id: str
    label: str
    family: str
    quality: CandidateQualitySummary
    operational: CandidateOperationalSummary
    bundle: CandidateBundleSummary
    source_artifacts: tuple[BenchmarkPackArtifactRef, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "family": self.family,
            "quality": self.quality.to_dict(),
            "operational": self.operational.to_dict(),
            "bundle": self.bundle.to_dict(),
            "source_artifacts": [artifact.to_dict() for artifact in self.source_artifacts],
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class PairwiseBenchmarkComparison:
    left_candidate_id: str
    right_candidate_id: str
    eer_delta_right_minus_left: float
    min_dcf_delta_right_minus_left: float
    latency_delta_ms_per_audio_right_minus_left: float | None
    process_rss_delta_mib_right_minus_left: float | None
    cuda_allocated_delta_mib_right_minus_left: float | None
    better_quality_candidate_id: str | None
    lower_latency_candidate_id: str | None
    lower_process_rss_candidate_id: str | None
    lower_cuda_allocated_candidate_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "left_candidate_id": self.left_candidate_id,
            "right_candidate_id": self.right_candidate_id,
            "eer_delta_right_minus_left": self.eer_delta_right_minus_left,
            "min_dcf_delta_right_minus_left": self.min_dcf_delta_right_minus_left,
            "latency_delta_ms_per_audio_right_minus_left": (
                self.latency_delta_ms_per_audio_right_minus_left
            ),
            "process_rss_delta_mib_right_minus_left": self.process_rss_delta_mib_right_minus_left,
            "cuda_allocated_delta_mib_right_minus_left": (
                self.cuda_allocated_delta_mib_right_minus_left
            ),
            "better_quality_candidate_id": self.better_quality_candidate_id,
            "lower_latency_candidate_id": self.lower_latency_candidate_id,
            "lower_process_rss_candidate_id": self.lower_process_rss_candidate_id,
            "lower_cuda_allocated_candidate_id": self.lower_cuda_allocated_candidate_id,
        }


@dataclass(frozen=True, slots=True)
class FinalBenchmarkPackSummary:
    candidate_count: int
    pairwise_comparison_count: int
    best_eer_candidate_id: str | None
    best_min_dcf_candidate_id: str | None
    lowest_latency_candidate_id: str | None
    lowest_process_rss_candidate_id: str | None
    lowest_cuda_allocated_candidate_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "pairwise_comparison_count": self.pairwise_comparison_count,
            "best_eer_candidate_id": self.best_eer_candidate_id,
            "best_min_dcf_candidate_id": self.best_min_dcf_candidate_id,
            "lowest_latency_candidate_id": self.lowest_latency_candidate_id,
            "lowest_process_rss_candidate_id": self.lowest_process_rss_candidate_id,
            "lowest_cuda_allocated_candidate_id": self.lowest_cuda_allocated_candidate_id,
        }


@dataclass(frozen=True, slots=True)
class FinalBenchmarkPackReport:
    title: str
    summary_text: str
    output_root: str
    source_config_artifact: BenchmarkPackArtifactRef | None
    notes: tuple[str, ...]
    summary: FinalBenchmarkPackSummary
    candidates: tuple[CandidateBenchmarkSummary, ...]
    pairwise_comparisons: tuple[PairwiseBenchmarkComparison, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "summary_text": self.summary_text,
            "output_root": self.output_root,
            "source_config_artifact": (
                None
                if self.source_config_artifact is None
                else self.source_config_artifact.to_dict()
            ),
            "notes": list(self.notes),
            "summary": self.summary.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "pairwise_comparisons": [
                comparison.to_dict() for comparison in self.pairwise_comparisons
            ],
        }


@dataclass(frozen=True, slots=True)
class WrittenFinalBenchmarkPack:
    output_root: str
    report_json_path: str
    report_markdown_path: str
    candidate_jsonl_path: str
    pairwise_jsonl_path: str
    summary: FinalBenchmarkPackSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "candidate_jsonl_path": self.candidate_jsonl_path,
            "pairwise_jsonl_path": self.pairwise_jsonl_path,
            "summary": self.summary.to_dict(),
        }


__all__ = [
    "BenchmarkPackArtifactRef",
    "CandidateBenchmarkSummary",
    "CandidateBundleSummary",
    "CandidateOperationalSummary",
    "CandidateQualitySummary",
    "FINAL_BENCHMARK_PACK_CANDIDATES_JSONL_NAME",
    "FINAL_BENCHMARK_PACK_JSON_NAME",
    "FINAL_BENCHMARK_PACK_MARKDOWN_NAME",
    "FINAL_BENCHMARK_PACK_PAIRWISE_JSONL_NAME",
    "FinalBenchmarkPackReport",
    "FinalBenchmarkPackSummary",
    "PairwiseBenchmarkComparison",
    "WrittenFinalBenchmarkPack",
]
