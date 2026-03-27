"""Repository-level system architecture contract for the speaker stack."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kryptonite.config import ProjectConfig
from kryptonite.serve.export_boundary import build_export_boundary_contract
from kryptonite.serve.inference_package import build_inference_package_contract

SYSTEM_ARCHITECTURE_FORMAT_VERSION = "kryptonite.system_architecture.v1"
SYSTEM_ARCHITECTURE_JSON_NAME = "system_architecture.json"
SYSTEM_ARCHITECTURE_MARKDOWN_NAME = "system_architecture.md"
DEFAULT_SYSTEM_ARCHITECTURE_OUTPUT_ROOT = "artifacts/system-architecture"


@dataclass(frozen=True, slots=True)
class ArchitectureStage:
    stage_id: str
    order: int
    status: str
    summary: str
    input_contract: str
    output_contract: str
    owner_modules: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModuleBoundary:
    module_id: str
    layer: str
    responsibility: str
    owned_paths: tuple[str, ...]
    public_entrypoints: tuple[str, ...]
    depends_on: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InterfacePoint:
    interface_id: str
    kind: str
    contract: str
    producer: str
    consumers: tuple[str, ...]
    stability: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LoggingPoint:
    point_id: str
    channel: str
    owner: str
    trigger: str
    payload_fields: tuple[str, ...]
    sinks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExportServePlacement:
    boundary_mode: str
    exported_subgraph: str
    runtime_pre_engine_steps: tuple[str, ...]
    runtime_post_engine_steps: tuple[str, ...]
    runtime_entrypoint: str
    http_entrypoint: str
    backend_fallback_chain: tuple[str, ...]
    validated_backends: dict[str, bool]
    health_endpoints: tuple[str, ...]
    metrics_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArchitectureArtifact:
    artifact_id: str
    required_for_handoff: bool
    path_hint: str
    producer: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArchitectureLimitation:
    limitation_id: str
    impact: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SystemArchitectureContract:
    title: str
    ticket_id: str
    decision_id: str
    format_version: str
    summary: str
    output_root: str
    pipeline_diagram: str
    canonical_stack: tuple[str, ...]
    stages: tuple[ArchitectureStage, ...]
    module_boundaries: tuple[ModuleBoundary, ...]
    interface_points: tuple[InterfacePoint, ...]
    logging_points: tuple[LoggingPoint, ...]
    export_and_serve: ExportServePlacement
    expected_artifacts: tuple[ArchitectureArtifact, ...]
    limitations: tuple[ArchitectureLimitation, ...]
    supporting_docs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "ticket_id": self.ticket_id,
            "decision_id": self.decision_id,
            "format_version": self.format_version,
            "summary": self.summary,
            "output_root": self.output_root,
            "pipeline_diagram": self.pipeline_diagram,
            "canonical_stack": list(self.canonical_stack),
            "stages": [item.to_dict() for item in self.stages],
            "module_boundaries": [item.to_dict() for item in self.module_boundaries],
            "interface_points": [item.to_dict() for item in self.interface_points],
            "logging_points": [item.to_dict() for item in self.logging_points],
            "export_and_serve": self.export_and_serve.to_dict(),
            "expected_artifacts": [item.to_dict() for item in self.expected_artifacts],
            "limitations": [item.to_dict() for item in self.limitations],
            "supporting_docs": list(self.supporting_docs),
        }


@dataclass(frozen=True, slots=True)
class WrittenSystemArchitectureContract:
    output_root: str
    report_json_path: str
    report_markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_system_architecture_contract(
    config: ProjectConfig,
    *,
    output_root: str = DEFAULT_SYSTEM_ARCHITECTURE_OUTPUT_ROOT,
    inferencer_backend: str = "feature_statistics",
    embedding_stage: str = "runtime",
    embedding_mode: str | None = "mean_std",
) -> SystemArchitectureContract:
    export_boundary = build_export_boundary_contract(
        config=config,
        inferencer_backend=inferencer_backend,
        embedding_stage=embedding_stage,
        embedding_mode=embedding_mode,
    )
    inference_package = build_inference_package_contract(onnx_model_file=None)
    metrics_path = config.telemetry.metrics_path if config.telemetry.enabled else None

    return SystemArchitectureContract(
        title="Kryptonite System Architecture v1",
        ticket_id="KVA-482",
        decision_id="kryptonite-2026-system-architecture-v1",
        format_version=SYSTEM_ARCHITECTURE_FORMAT_VERSION,
        summary=(
            "One repository-level architecture contract now ties together data loading, shared "
            "feature frontend, encoder boundary, cosine scoring, offline normalization and "
            "calibration, plus the thin serving/export surfaces that consume the same contracts."
        ),
        output_root=output_root,
        pipeline_diagram=(
            f"{config.normalization.target_sample_rate_hz} Hz mono audio -> optional loudness/VAD "
            f"-> chunking -> {config.features.num_mel_bins}-bin log-Mel/Fbank -> "
            f"{export_boundary.input_tensor.name} -> compact encoder or "
            f"{inferencer_backend} fallback -> {export_boundary.output_tensor.name} -> cosine "
            "scoring -> optional AS-norm/TAS-norm offline -> threshold/calibration -> local/HTTP "
            "serve surfaces"
        ),
        canonical_stack=(
            (
                f"Raw audio is normalized to {config.normalization.target_sample_rate_hz} Hz, "
                f"{config.normalization.target_channels} channel(s), "
                f"{config.normalization.output_format}/"
                f"PCM{config.normalization.output_pcm_bits_per_sample}."
            ),
            (
                f"Frontend policy keeps loudness=`{config.normalization.loudness_mode}` and "
                f"VAD=`{config.vad.mode}` under the shared runtime/data contract."
            ),
            (
                f"Chunking and features stay outside the exported graph and produce "
                f"`{config.features.num_mel_bins}`-bin log-Mel/Fbank tensors."
            ),
            (
                f"Encoder boundary stays `{export_boundary.boundary}` from "
                f"`{export_boundary.input_tensor.name}` to "
                f"`{export_boundary.output_tensor.name}`."
            ),
            "Shared cosine scoring is the only first-class runtime scoring primitive.",
            "AS-norm and TAS-norm are offline evaluation/experiment layers, not live HTTP logic.",
            (
                "Threshold calibration and verification reports remain the handoff artifacts "
                "for decisions."
            ),
            "Serve adapters stay thin and reuse Inferencer plus structured telemetry.",
        ),
        stages=(
            ArchitectureStage(
                stage_id="raw_audio_ingest",
                order=1,
                status="implemented",
                summary=(
                    "Shared audio loading resolves paths, decode, mono fold-down, resampling, "
                    "bounded loudness normalization, and optional VAD trimming."
                ),
                input_contract="audio path, manifest row, or HTTP/local request payload",
                output_contract="normalized waveform plus trim/loudness metadata",
                owner_modules=(
                    "src/kryptonite/data/audio_loader.py",
                    "src/kryptonite/data/audio_io.py",
                    "src/kryptonite/data/loudness.py",
                    "src/kryptonite/data/vad.py",
                    "src/kryptonite/data/normalization/",
                ),
                notes=(
                    (
                        "The same AudioLoadRequest contract feeds both baseline training "
                        "exports and serving."
                    ),
                ),
            ),
            ArchitectureStage(
                stage_id="feature_frontend",
                order=2,
                status="implemented",
                summary=(
                    "Waveforms are chunked and converted into log-Mel/Fbank frames with optional "
                    "cache/report helpers around the same frontend."
                ),
                input_contract="normalized waveform batches and chunking config",
                output_contract=_render_tensor_signature(export_boundary.input_tensor),
                owner_modules=(
                    "src/kryptonite/features/chunking.py",
                    "src/kryptonite/features/fbank.py",
                    "src/kryptonite/features/cache.py",
                    "src/kryptonite/features/reporting.py",
                ),
                notes=(
                    (
                        "This stage stays runtime-owned even when later ONNX/TensorRT exports "
                        "are used."
                    ),
                ),
            ),
            ArchitectureStage(
                stage_id="encoder_runtime",
                order=3,
                status="implemented_with_runtime_fallback",
                summary=(
                    "Compact speaker encoders define the intended learned boundary, while the "
                    "checked-in runtime currently falls back to the feature_statistics backend."
                ),
                input_contract=_render_tensor_signature(export_boundary.input_tensor),
                output_contract=_render_tensor_signature(export_boundary.output_tensor),
                owner_modules=(
                    "src/kryptonite/models/campp/model.py",
                    "src/kryptonite/models/eres2netv2/model.py",
                    "src/kryptonite/serve/inference_backend.py",
                ),
                notes=(
                    "The export path is ready for encoder-only graphs, but the default local demo "
                    "runtime still proves contract shape through feature_statistics.",
                ),
            ),
            ArchitectureStage(
                stage_id="enrollment_and_cosine_scoring",
                order=4,
                status="implemented",
                summary=(
                    "Embeddings are pooled into enrollment centroids and scored through the "
                    "shared cosine scorer for pairwise and one-to-many flows."
                ),
                input_contract="embedding batches or enrollment/probe embedding matrices",
                output_contract="cosine score vectors, score matrices, and enrollment centroids",
                owner_modules=(
                    "src/kryptonite/models/scoring.py",
                    "src/kryptonite/serve/scoring_service.py",
                    "src/kryptonite/serve/enrollment_cache.py",
                ),
                notes=(
                    (
                        "This is the only live runtime scoring path today and is shared with "
                        "offline evaluation."
                    ),
                ),
            ),
            ArchitectureStage(
                stage_id="score_normalization",
                order=5,
                status="offline_experimental",
                summary=(
                    "AS-norm and TAS-norm reuse exported embeddings plus cohort banks to build "
                    "offline score-normalization experiments."
                ),
                input_contract="raw verification scores, embedding exports, metadata, cohort bank",
                output_contract="normalized verification score rows and experiment summaries",
                owner_modules=(
                    "src/kryptonite/eval/score_normalization.py",
                    "src/kryptonite/eval/as_norm.py",
                    "src/kryptonite/eval/tas_norm.py",
                    "src/kryptonite/eval/tas_norm_experiment.py",
                ),
                notes=(
                    "These paths are explicitly outside the current HTTP/runtime serving surface.",
                ),
            ),
            ArchitectureStage(
                stage_id="evaluation_and_calibration",
                order=6,
                status="implemented",
                summary=(
                    "Verification protocol builders, reports, slice analysis, and threshold "
                    "calibration turn raw scores into decision-ready offline artifacts."
                ),
                input_contract=(
                    "trial manifests, score rows, metadata slices, candidate bundle artifacts"
                ),
                output_contract=(
                    "verification reports, dashboards, threshold bundles, error analysis"
                ),
                owner_modules=(
                    "src/kryptonite/eval/verification_protocol.py",
                    "src/kryptonite/eval/verification_report.py",
                    "src/kryptonite/eval/verification_threshold_calibration.py",
                    "src/kryptonite/eval/verification_error_analysis/",
                ),
                notes=(
                    (
                        "Verification is the first-class task; identification remains a "
                        "compatibility mode."
                    ),
                ),
            ),
            ArchitectureStage(
                stage_id="serve_and_transport",
                order=7,
                status="implemented",
                summary=(
                    "Local callers, FastAPI transport, demo endpoints, and telemetry all delegate "
                    "to the same Inferencer-owned runtime contract."
                ),
                input_contract="audio paths or embeddings over local Python and JSON HTTP payloads",
                output_contract="health metadata, embeddings, enrollments, scores, demo decisions",
                owner_modules=(
                    "src/kryptonite/serve/inferencer.py",
                    "src/kryptonite/serve/http.py",
                    "src/kryptonite/serve/api_models.py",
                    "src/kryptonite/serve/telemetry.py",
                ),
                notes=(
                    (
                        "Transport stays thin; core runtime logic and validation live in "
                        "src/kryptonite/serve."
                    ),
                ),
            ),
        ),
        module_boundaries=(
            ModuleBoundary(
                module_id="data_contracts",
                layer="data",
                responsibility=(
                    "Dataset manifests, audio ingestion, normalization policy, VAD, and "
                    "verification trial materialization."
                ),
                owned_paths=(
                    "src/kryptonite/data/audio_loader.py",
                    "src/kryptonite/data/normalization/",
                    "src/kryptonite/data/vad.py",
                    "src/kryptonite/data/verification_trials.py",
                ),
                public_entrypoints=(
                    "kryptonite.data.AudioLoadRequest",
                    "kryptonite.data.load_audio",
                    "kryptonite.data.load_manifest_audio",
                ),
                depends_on=("configs/base.toml",),
            ),
            ModuleBoundary(
                module_id="feature_frontend",
                layer="features",
                responsibility=(
                    "Chunking, Fbank extraction, optional feature cache, and frontend reporting."
                ),
                owned_paths=(
                    "src/kryptonite/features/chunking.py",
                    "src/kryptonite/features/fbank.py",
                    "src/kryptonite/features/cache.py",
                ),
                public_entrypoints=(
                    "kryptonite.features.UtteranceChunkingRequest",
                    "kryptonite.features.FbankExtractionRequest",
                    "kryptonite.features.FbankExtractor",
                ),
                depends_on=("data_contracts",),
            ),
            ModuleBoundary(
                module_id="model_family",
                layer="models",
                responsibility=(
                    "Speaker-encoder definitions plus the shared embedding/cosine scoring helpers."
                ),
                owned_paths=(
                    "src/kryptonite/models/campp/",
                    "src/kryptonite/models/eres2netv2/",
                    "src/kryptonite/models/scoring.py",
                ),
                public_entrypoints=(
                    "kryptonite.models.campp",
                    "kryptonite.models.eres2netv2",
                    "kryptonite.models.cosine_score_pairs",
                ),
                depends_on=("feature_frontend",),
            ),
            ModuleBoundary(
                module_id="training_recipes",
                layer="training",
                responsibility=(
                    "Manifest-backed baseline training, augmentation runtime, "
                    "optimization runtime, and stage-specific CAM++/ERes2NetV2 pipelines."
                ),
                owned_paths=(
                    "src/kryptonite/training/speaker_baseline.py",
                    "src/kryptonite/training/campp/",
                    "src/kryptonite/training/eres2netv2/",
                    "src/kryptonite/training/augmentation_runtime.py",
                ),
                public_entrypoints=(
                    "kryptonite.training.speaker_baseline",
                    "kryptonite.training.campp.pipeline",
                    "kryptonite.training.eres2netv2.pipeline",
                ),
                depends_on=("data_contracts", "feature_frontend", "model_family"),
            ),
            ModuleBoundary(
                module_id="evaluation_suite",
                layer="eval",
                responsibility=(
                    "Verification protocol assembly, score normalization experiments, "
                    "slice analysis, threshold calibration, and release-oriented reports."
                ),
                owned_paths=(
                    "src/kryptonite/eval/verification_protocol.py",
                    "src/kryptonite/eval/verification_report.py",
                    "src/kryptonite/eval/verification_threshold_calibration.py",
                    "src/kryptonite/eval/as_norm.py",
                    "src/kryptonite/eval/tas_norm.py",
                ),
                public_entrypoints=(
                    "kryptonite.eval.verification_protocol",
                    "kryptonite.eval.verification_report",
                    "kryptonite.eval.verification_threshold_calibration",
                ),
                depends_on=("training_recipes", "model_family"),
            ),
            ModuleBoundary(
                module_id="serving_runtime",
                layer="serve",
                responsibility=(
                    "Unified inferencer, enrollment/scoring state, JSON HTTP transport, "
                    "demo flows, and runtime observability."
                ),
                owned_paths=(
                    "src/kryptonite/serve/inferencer.py",
                    "src/kryptonite/serve/scoring_service.py",
                    "src/kryptonite/serve/http.py",
                    "src/kryptonite/serve/telemetry.py",
                ),
                public_entrypoints=(
                    "kryptonite.serve.Inferencer",
                    "kryptonite.serve.http.create_http_app",
                    "kryptonite.serve.telemetry.ServiceTelemetry",
                ),
                depends_on=(
                    "data_contracts",
                    "feature_frontend",
                    "model_family",
                    "evaluation_suite",
                ),
            ),
            ModuleBoundary(
                module_id="export_and_deployment",
                layer="deploy",
                responsibility=(
                    "Export boundary metadata, backend selection contract, model-bundle packaging, "
                    "and Triton/deployment handoff builders."
                ),
                owned_paths=(
                    "src/kryptonite/serve/export_boundary.py",
                    "src/kryptonite/serve/inference_package.py",
                    "src/kryptonite/serve/triton_repository.py",
                    "src/kryptonite/serve/submission_bundle.py",
                ),
                public_entrypoints=(
                    "kryptonite.serve.export_boundary.build_export_boundary_contract",
                    "kryptonite.serve.inference_package.build_inference_package_contract",
                    "kryptonite.serve.triton_repository.build_triton_model_repository",
                ),
                depends_on=("serving_runtime", "evaluation_suite"),
            ),
        ),
        interface_points=(
            InterfacePoint(
                interface_id="audio_load_request",
                kind="typed_python_contract",
                contract=(
                    "AudioLoadRequest carries sample-rate/channel/normalization/VAD rules into "
                    "training and serving loaders."
                ),
                producer="src/kryptonite/data/audio_loader.py",
                consumers=(
                    "src/kryptonite/training/speaker_baseline.py",
                    "src/kryptonite/serve/inferencer.py",
                ),
                stability="first_class",
            ),
            InterfacePoint(
                interface_id="fbank_frontend_request",
                kind="typed_python_contract",
                contract=(
                    "UtteranceChunkingRequest plus FbankExtractionRequest define the shared "
                    "frontend before any encoder boundary."
                ),
                producer="src/kryptonite/features/chunking.py + src/kryptonite/features/fbank.py",
                consumers=(
                    "src/kryptonite/training/speaker_baseline.py",
                    "src/kryptonite/serve/inference_backend.py",
                ),
                stability="first_class",
            ),
            InterfacePoint(
                interface_id="encoder_export_boundary",
                kind="machine_readable_contract",
                contract=(
                    f"{_render_tensor_signature(export_boundary.input_tensor)} -> "
                    f"{_render_tensor_signature(export_boundary.output_tensor)}"
                ),
                producer="src/kryptonite/serve/export_boundary.py",
                consumers=(
                    "src/kryptonite/serve/inferencer.py",
                    "src/kryptonite/serve/triton_repository.py",
                    "future ONNX/TensorRT export tasks",
                ),
                stability="first_class",
            ),
            InterfacePoint(
                interface_id="cosine_scoring_api",
                kind="shared_python_api",
                contract=(
                    "Normalized pairwise and one-to-many scoring over embedding matrices, plus "
                    "enrollment centroid pooling."
                ),
                producer="src/kryptonite/models/scoring.py",
                consumers=(
                    "src/kryptonite/training/speaker_baseline.py",
                    "src/kryptonite/serve/scoring_service.py",
                    "src/kryptonite/eval/score_normalization.py",
                ),
                stability="first_class",
            ),
            InterfacePoint(
                interface_id="inferencer_runtime_api",
                kind="shared_python_api",
                contract=(
                    "Inferencer owns raw-audio embed/enroll/verify/benchmark flows and exposes one "
                    "runtime surface for local code and HTTP."
                ),
                producer="src/kryptonite/serve/inferencer.py",
                consumers=(
                    "src/kryptonite/serve/http.py",
                    "deploy/demo smoke checks",
                    "local Python callers",
                ),
                stability="first_class",
            ),
            InterfacePoint(
                interface_id="http_transport_surface",
                kind="json_http_api",
                contract=(
                    "FastAPI exposes /health, /metrics, /embed, /enroll, /verify, /score/*, and "
                    "demo endpoints on top of the Inferencer."
                ),
                producer="src/kryptonite/serve/http.py",
                consumers=("browser demo", "integration smoke tests", "external callers"),
                stability="thin_adapter",
            ),
        ),
        logging_points=(
            LoggingPoint(
                point_id="local_training_tracker",
                channel="artifact_tracking",
                owner="src/kryptonite/tracking.py",
                trigger="train/eval scripts start a LocalTracker run",
                payload_fields=(
                    "run.json",
                    "params.json",
                    "metrics.jsonl",
                    "artifacts.json",
                ),
                sinks=(config.tracking.output_root, "copied artifact files when enabled"),
            ),
            LoggingPoint(
                point_id="verification_reports",
                channel="structured_artifacts",
                owner="src/kryptonite/eval/verification_report.py",
                trigger="offline score evaluation and threshold calibration complete",
                payload_fields=(
                    "verification_eval_report.json",
                    "verification_threshold_calibration.json",
                    "slice breakdowns",
                ),
                sinks=("artifacts/**", "docs/ for curated summaries"),
            ),
            LoggingPoint(
                point_id="serve_json_logs_and_metrics",
                channel="runtime_observability",
                owner="src/kryptonite/serve/telemetry.py",
                trigger="service start, HTTP requests, validation errors, inference operations",
                payload_fields=(
                    "service",
                    "backend",
                    "implementation",
                    "model_version",
                    "latency_ms",
                    "audio_count",
                    "total_chunk_count",
                ),
                sinks=(
                    "stdout JSON logs",
                    metrics_path or "<telemetry disabled>",
                ),
            ),
            LoggingPoint(
                point_id="health_runtime_metadata",
                channel="runtime_metadata",
                owner="src/kryptonite/serve/inferencer.py",
                trigger="GET /health or local health payload inspection",
                payload_fields=(
                    "selected backend",
                    "model bundle metadata",
                    "export boundary summary",
                    "telemetry summary",
                    "enrollment cache summary",
                ),
                sinks=("/health", "/healthz", "/readyz"),
            ),
        ),
        export_and_serve=ExportServePlacement(
            boundary_mode=export_boundary.boundary,
            exported_subgraph=(
                f"{export_boundary.input_tensor.name} -> encoder forward -> "
                f"{export_boundary.output_tensor.name}"
            ),
            runtime_pre_engine_steps=export_boundary.runtime_pre_engine_steps,
            runtime_post_engine_steps=export_boundary.runtime_post_engine_steps,
            runtime_entrypoint="src/kryptonite/serve/inferencer.py::Inferencer.from_config",
            http_entrypoint="src/kryptonite/serve/http.py::create_http_app",
            backend_fallback_chain=inference_package.backend_chain,
            validated_backends=dict(inference_package.validated_backends),
            health_endpoints=("/health", "/healthz", "/readyz"),
            metrics_path=metrics_path,
        ),
        expected_artifacts=(
            ArchitectureArtifact(
                artifact_id="architecture_adr",
                required_for_handoff=True,
                path_hint="docs/system-architecture-v1.md",
                producer="checked-in repository documentation",
                description=(
                    "Human-readable architecture note that fixes the pipeline diagram, module "
                    "boundaries, interfaces, and observability points."
                ),
            ),
            ArchitectureArtifact(
                artifact_id="architecture_snapshot_json",
                required_for_handoff=True,
                path_hint=f"{output_root}/{SYSTEM_ARCHITECTURE_JSON_NAME}",
                producer="scripts/build_system_architecture.py",
                description=(
                    "Machine-readable architecture snapshot for downstream tooling and handoff."
                ),
            ),
            ArchitectureArtifact(
                artifact_id="architecture_snapshot_markdown",
                required_for_handoff=True,
                path_hint=f"{output_root}/{SYSTEM_ARCHITECTURE_MARKDOWN_NAME}",
                producer="scripts/build_system_architecture.py",
                description="Generated markdown companion to the architecture JSON snapshot.",
            ),
            ArchitectureArtifact(
                artifact_id="export_boundary_contract",
                required_for_handoff=True,
                path_hint="artifacts/export-boundary/export_boundary.json",
                producer="scripts/export_boundary_report.py",
                description=(
                    "Machine-readable encoder boundary consumed by runtime and export tasks."
                ),
            ),
            ArchitectureArtifact(
                artifact_id="model_task_contract",
                required_for_handoff=True,
                path_hint="artifacts/model-task-contract/model_task_contract.json",
                producer="scripts/build_model_task_contract.py",
                description=(
                    "Task-level contract that keeps the architecture aligned with verification."
                ),
            ),
        ),
        limitations=(
            ArchitectureLimitation(
                limitation_id="runtime_backend_is_not_final_encoder",
                impact="serve_path",
                description=(
                    "The checked-in demo/runtime path still uses `feature_statistics`, so "
                    "serving currently validates contracts and integration points more than "
                    "final model quality."
                ),
            ),
            ArchitectureLimitation(
                limitation_id="score_normalization_is_offline_only",
                impact="scoring_surface",
                description=(
                    "AS-norm and TAS-norm live in offline evaluation/experiment modules "
                    "and are not wired into the HTTP runtime path yet."
                ),
            ),
            ArchitectureLimitation(
                limitation_id="encoder_only_export_boundary",
                impact="export_scope",
                description=(
                    "Waveform decode, loudness, VAD, chunking, and Fbank extraction "
                    "intentionally stay outside exported ONNX/TensorRT graphs."
                ),
            ),
            ArchitectureLimitation(
                limitation_id="tracking_is_local_first",
                impact="observability_scope",
                description=(
                    "The repository standardizes on a local tracker today; MLflow/W&B "
                    "remain future adapter targets rather than active first-class backends."
                ),
            ),
        ),
        supporting_docs=(
            "docs/model-task-contract.md",
            "docs/audio-loader.md",
            "docs/audio-fbank-extraction.md",
            "docs/embedding-scoring.md",
            "docs/evaluation-package.md",
            "docs/threshold-calibration.md",
            "docs/export-boundary.md",
            "docs/unified-inference-wrapper.md",
            "docs/inference-observability.md",
            "docs/tracking.md",
        ),
    )


def render_system_architecture_markdown(contract: SystemArchitectureContract) -> str:
    backend_chain = ", ".join(contract.export_and_serve.backend_fallback_chain)
    pre_engine_steps = ", ".join(contract.export_and_serve.runtime_pre_engine_steps)
    post_engine_steps = ", ".join(contract.export_and_serve.runtime_post_engine_steps)
    lines = [
        f"# {contract.title}",
        "",
        f"`{contract.ticket_id}` fixes one explicit repository architecture contract for the "
        "Kryptonite speaker-recognition stack.",
        "",
        "## Decision",
        "",
        f"- format version: `{contract.format_version}`",
        (
            f"- current runtime backend: `{contract.export_and_serve.validated_backends}` "
            f"validated over `{backend_chain}`"
        ),
        f"- export boundary mode: `{contract.export_and_serve.boundary_mode}`",
        f"- health endpoints: `{', '.join(contract.export_and_serve.health_endpoints)}`",
        "",
        contract.summary,
        "",
        "## Pipeline Diagram",
        "",
        "```text",
        contract.pipeline_diagram,
        "```",
        "",
        "## Canonical Stack",
        "",
    ]
    for item in contract.canonical_stack:
        lines.append(f"- {item}")

    lines.extend(["", "## Pipeline Stages", ""])
    for stage in contract.stages:
        lines.extend(
            [
                f"### {stage.order}. {stage.stage_id}",
                "",
                f"- status: `{stage.status}`",
                f"- summary: {stage.summary}",
                f"- input contract: {stage.input_contract}",
                f"- output contract: {stage.output_contract}",
                f"- owner modules: `{', '.join(stage.owner_modules)}`",
            ]
        )
        if stage.notes:
            lines.append(f"- notes: {' '.join(stage.notes)}")
        lines.append("")

    lines.extend(["## Module Boundaries", ""])
    for boundary in contract.module_boundaries:
        lines.extend(
            [
                f"### {boundary.module_id}",
                "",
                f"- layer: `{boundary.layer}`",
                f"- responsibility: {boundary.responsibility}",
                f"- owned paths: `{', '.join(boundary.owned_paths)}`",
                f"- public entrypoints: `{', '.join(boundary.public_entrypoints)}`",
                f"- depends on: `{', '.join(boundary.depends_on)}`",
                "",
            ]
        )

    lines.extend(["## Interfaces", ""])
    for interface in contract.interface_points:
        lines.extend(
            [
                f"### {interface.interface_id}",
                "",
                f"- kind: `{interface.kind}`",
                f"- contract: {interface.contract}",
                f"- producer: `{interface.producer}`",
                f"- consumers: `{', '.join(interface.consumers)}`",
                f"- stability: `{interface.stability}`",
                "",
            ]
        )

    lines.extend(["## Logging Points", ""])
    for point in contract.logging_points:
        lines.extend(
            [
                f"### {point.point_id}",
                "",
                f"- channel: `{point.channel}`",
                f"- owner: `{point.owner}`",
                f"- trigger: {point.trigger}",
                f"- payload fields: `{', '.join(point.payload_fields)}`",
                f"- sinks: `{', '.join(point.sinks)}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Export And Serve Placement",
            "",
            f"- exported subgraph: `{contract.export_and_serve.exported_subgraph}`",
            f"- runtime pre-engine steps: `{pre_engine_steps}`",
            f"- runtime post-engine steps: `{post_engine_steps}`",
            f"- runtime entrypoint: `{contract.export_and_serve.runtime_entrypoint}`",
            f"- HTTP entrypoint: `{contract.export_and_serve.http_entrypoint}`",
            f"- backend fallback chain: `{backend_chain}`",
            f"- validated backends: `{contract.export_and_serve.validated_backends}`",
            f"- metrics path: `{contract.export_and_serve.metrics_path}`",
            "",
            "## Expected Artifacts",
            "",
        ]
    )
    for artifact in contract.expected_artifacts:
        lines.extend(
            [
                f"### {artifact.artifact_id}",
                "",
                f"- required for handoff: `{artifact.required_for_handoff}`",
                f"- path hint: `{artifact.path_hint}`",
                f"- producer: `{artifact.producer}`",
                f"- description: {artifact.description}",
                "",
            ]
        )

    lines.extend(["## Limitations", ""])
    for limitation in contract.limitations:
        lines.append(
            f"- `{limitation.limitation_id}` ({limitation.impact}): {limitation.description}"
        )

    lines.extend(["", "## Supporting References", ""])
    lines.extend(f"- `{path}`" for path in contract.supporting_docs)
    lines.extend(
        [
            "",
            "## Rebuild",
            "",
            "```bash",
            "uv run python scripts/build_system_architecture.py --config configs/base.toml",
            "```",
            "",
            f"This writes `{contract.output_root}/{SYSTEM_ARCHITECTURE_JSON_NAME}` and "
            f"`{contract.output_root}/{SYSTEM_ARCHITECTURE_MARKDOWN_NAME}`.",
        ]
    )
    return "\n".join(lines)


def write_system_architecture_contract(
    contract: SystemArchitectureContract,
    *,
    project_root: Path | str = ".",
) -> WrittenSystemArchitectureContract:
    root = Path(project_root)
    output_root = root / contract.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    report_json_path = output_root / SYSTEM_ARCHITECTURE_JSON_NAME
    report_markdown_path = output_root / SYSTEM_ARCHITECTURE_MARKDOWN_NAME
    report_json_path.write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_markdown_path.write_text(
        render_system_architecture_markdown(contract) + "\n",
        encoding="utf-8",
    )
    return WrittenSystemArchitectureContract(
        output_root=str(output_root),
        report_json_path=str(report_json_path),
        report_markdown_path=str(report_markdown_path),
    )


def _render_tensor_signature(tensor: Any) -> str:
    axes = ", ".join(f"{axis.name}={axis.size}" for axis in tensor.axes)
    return f"{tensor.name} ({tensor.layout}, {tensor.dtype}; {axes})"


__all__ = [
    "ArchitectureArtifact",
    "ArchitectureLimitation",
    "ArchitectureStage",
    "DEFAULT_SYSTEM_ARCHITECTURE_OUTPUT_ROOT",
    "ExportServePlacement",
    "InterfacePoint",
    "LoggingPoint",
    "ModuleBoundary",
    "SYSTEM_ARCHITECTURE_FORMAT_VERSION",
    "SYSTEM_ARCHITECTURE_JSON_NAME",
    "SYSTEM_ARCHITECTURE_MARKDOWN_NAME",
    "SystemArchitectureContract",
    "WrittenSystemArchitectureContract",
    "build_system_architecture_contract",
    "render_system_architecture_markdown",
    "write_system_architecture_contract",
]
