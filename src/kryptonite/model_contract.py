"""Repository-level task and model contract for speaker verification."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kryptonite.config import ProjectConfig
from kryptonite.serve.export_boundary import build_export_boundary_contract

MODEL_TASK_CONTRACT_FORMAT_VERSION = "kryptonite.model_contract.v1"
MODEL_TASK_CONTRACT_JSON_NAME = "model_task_contract.json"
MODEL_TASK_CONTRACT_MARKDOWN_NAME = "model_task_contract.md"
DEFAULT_MODEL_TASK_CONTRACT_OUTPUT_ROOT = "artifacts/model-task-contract"


@dataclass(frozen=True, slots=True)
class RawAudioContract:
    sample_rate_hz: int
    channels: int
    output_format: str
    pcm_bits_per_sample: int
    loudness_mode: str
    vad_mode: str
    train_crop_seconds: tuple[float, float]
    eval_chunk_seconds: float
    eval_chunk_overlap_seconds: float
    demo_chunk_seconds: float
    demo_chunk_overlap_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EmbeddingContract:
    boundary_mode: str
    frontend_location: str
    inferencer_backend: str
    embedding_stage: str
    embedding_mode: str | None
    input_signature: str
    output_signature: str
    output_embedding_dim: int | str
    scoring_metric: str
    enrollment_pooling: tuple[str, ...]
    runtime_pre_engine_steps: tuple[str, ...]
    runtime_post_engine_steps: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TaskModeContract:
    mode_id: str
    support_level: str
    summary: str
    request_unit: str
    score_output: str
    decision_output: str
    metric_focus: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrialTypeContract:
    trial_type_id: str
    mode_id: str
    support_level: str
    description: str
    unit_of_evaluation: str
    required_fields: tuple[str, ...]
    label_space: tuple[str, ...]
    slice_fields: tuple[str, ...] = ()
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArtifactContract:
    artifact_id: str
    required_for_handoff: bool
    path_hint: str
    producer: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ContractLimitation:
    limitation_id: str
    impact: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelTaskContract:
    title: str
    ticket_id: str
    decision_id: str
    format_version: str
    summary: str
    output_root: str
    primary_task_mode: str
    canonical_workflow: tuple[str, ...]
    raw_audio_contract: RawAudioContract
    embedding_contract: EmbeddingContract
    task_modes: tuple[TaskModeContract, ...]
    trial_types: tuple[TrialTypeContract, ...]
    expected_artifacts: tuple[ArtifactContract, ...]
    limitations: tuple[ContractLimitation, ...]
    supporting_docs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "ticket_id": self.ticket_id,
            "decision_id": self.decision_id,
            "format_version": self.format_version,
            "summary": self.summary,
            "output_root": self.output_root,
            "primary_task_mode": self.primary_task_mode,
            "canonical_workflow": list(self.canonical_workflow),
            "raw_audio_contract": self.raw_audio_contract.to_dict(),
            "embedding_contract": self.embedding_contract.to_dict(),
            "task_modes": [item.to_dict() for item in self.task_modes],
            "trial_types": [item.to_dict() for item in self.trial_types],
            "expected_artifacts": [item.to_dict() for item in self.expected_artifacts],
            "limitations": [item.to_dict() for item in self.limitations],
            "supporting_docs": list(self.supporting_docs),
        }


@dataclass(frozen=True, slots=True)
class WrittenModelTaskContract:
    output_root: str
    report_json_path: str
    report_markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_model_task_contract(
    config: ProjectConfig,
    *,
    output_root: str = DEFAULT_MODEL_TASK_CONTRACT_OUTPUT_ROOT,
    inferencer_backend: str = "feature_statistics",
    embedding_stage: str = "runtime",
    embedding_mode: str | None = "mean_std",
) -> ModelTaskContract:
    export_boundary = build_export_boundary_contract(
        config=config,
        inferencer_backend=inferencer_backend,
        embedding_stage=embedding_stage,
        embedding_mode=embedding_mode,
    )

    return ModelTaskContract(
        title="Kryptonite Speaker Model Task Contract",
        ticket_id="KVA-480",
        decision_id="kryptonite-2026-model-task-contract",
        format_version=MODEL_TASK_CONTRACT_FORMAT_VERSION,
        summary=(
            "Primary repository task is speaker verification over enrolled references and probe "
            "audio. The same embedding space must remain compatible with closed-set and open-set "
            "identification, but verification is the only first-class task contract today."
        ),
        output_root=output_root,
        primary_task_mode="verification",
        canonical_workflow=(
            "enrollment audio -> normalized embedding centroid",
            "probe audio -> probe embedding",
            "cosine score between probe and enrolled speaker reference",
            "threshold profile -> verification decision",
        ),
        raw_audio_contract=RawAudioContract(
            sample_rate_hz=config.normalization.target_sample_rate_hz,
            channels=config.normalization.target_channels,
            output_format=config.normalization.output_format,
            pcm_bits_per_sample=config.normalization.output_pcm_bits_per_sample,
            loudness_mode=config.normalization.loudness_mode,
            vad_mode=config.vad.mode,
            train_crop_seconds=(
                config.chunking.train_min_crop_seconds,
                config.chunking.train_max_crop_seconds,
            ),
            eval_chunk_seconds=config.chunking.eval_chunk_seconds,
            eval_chunk_overlap_seconds=config.chunking.eval_chunk_overlap_seconds,
            demo_chunk_seconds=config.chunking.demo_chunk_seconds,
            demo_chunk_overlap_seconds=config.chunking.demo_chunk_overlap_seconds,
        ),
        embedding_contract=EmbeddingContract(
            boundary_mode=export_boundary.boundary,
            frontend_location=export_boundary.frontend_location,
            inferencer_backend=export_boundary.inferencer_backend,
            embedding_stage=export_boundary.embedding_stage,
            embedding_mode=export_boundary.embedding_mode,
            input_signature=_render_tensor_signature(export_boundary.input_tensor),
            output_signature=_render_tensor_signature(export_boundary.output_tensor),
            output_embedding_dim=export_boundary.output_tensor.axes[-1].size,
            scoring_metric="cosine_similarity",
            enrollment_pooling=(
                "l2_normalize_each_embedding",
                "average_embeddings",
                "l2_normalize_enrollment_centroid",
            ),
            runtime_pre_engine_steps=export_boundary.runtime_pre_engine_steps,
            runtime_post_engine_steps=export_boundary.runtime_post_engine_steps,
        ),
        task_modes=(
            TaskModeContract(
                mode_id="verification",
                support_level="primary_implemented",
                summary=(
                    "Canonical flow: enroll speaker references, embed probe audio, compute cosine "
                    "score, and optionally apply a thresholded accept/reject decision."
                ),
                request_unit="enrollment_id + probe audio or embeddings",
                score_output="single cosine score per enrollment/probe comparison",
                decision_output="boolean accept/reject after threshold selection",
                metric_focus=("EER", "minDCF", "slice breakdown", "error analysis"),
                notes=(
                    "Implemented in the raw-audio Inferencer, FastAPI adapter, and offline score "
                    "evaluation stack.",
                ),
            ),
            TaskModeContract(
                mode_id="closed_set_identification",
                support_level="compatible_not_first_class",
                summary=(
                    "Probe audio is ranked against a finite gallery of enrolled speakers via the "
                    "same embedding space and cosine score matrix."
                ),
                request_unit="probe audio + finite enrolled speaker gallery",
                score_output="ranked cosine score list over all enrolled speaker references",
                decision_output="top-1/top-k predicted speaker id",
                metric_focus=("top-1 accuracy", "top-k accuracy", "confusion analysis"),
                notes=(
                    "The shared scorer already supports one-to-many ranking, but there is no "
                    "dedicated repo-native identification report or serving endpoint yet.",
                ),
            ),
            TaskModeContract(
                mode_id="open_set_identification",
                support_level="compatible_not_first_class",
                summary=(
                    "Probe audio is ranked against a gallery, then rejected when the best score "
                    "stays below the active decision threshold."
                ),
                request_unit="probe audio + finite gallery + reject threshold",
                score_output="ranked cosine score list plus best-match score",
                decision_output="predicted speaker id or `unknown` reject",
                metric_focus=("open-set recall", "false accept rate", "false reject rate"),
                notes=(
                    "Open-set identification remains a compatibility target built on the same "
                    "encoder and thresholding contract as verification.",
                ),
            ),
        ),
        trial_types=(
            TrialTypeContract(
                trial_type_id="verification_pair",
                mode_id="verification",
                support_level="implemented",
                description=("Pairwise same-speaker / different-speaker verification trial."),
                unit_of_evaluation="left_audio x right_audio pair",
                required_fields=(
                    "left_audio",
                    "right_audio",
                    "label",
                    "left_speaker_id",
                    "right_speaker_id",
                ),
                label_space=("positive", "negative"),
                slice_fields=(
                    "duration_bucket",
                    "domain_relation",
                    "channel_relation",
                ),
                source_path="src/kryptonite/data/verification_trials.py",
            ),
            TrialTypeContract(
                trial_type_id="closed_set_gallery_probe",
                mode_id="closed_set_identification",
                support_level="planned",
                description=(
                    "Probe utterance scored against a closed gallery of enrolled speakers."
                ),
                unit_of_evaluation="probe audio against finite candidate gallery",
                required_fields=("probe_audio", "candidate_enrollment_ids", "expected_speaker_id"),
                label_space=("known_speaker_id",),
            ),
            TrialTypeContract(
                trial_type_id="open_set_gallery_probe",
                mode_id="open_set_identification",
                support_level="planned",
                description=(
                    "Probe utterance scored against a gallery with the option to reject as "
                    "`unknown` when no speaker crosses threshold."
                ),
                unit_of_evaluation="probe audio against finite gallery plus reject option",
                required_fields=(
                    "probe_audio",
                    "candidate_enrollment_ids",
                    "expected_speaker_id_or_unknown",
                ),
                label_space=("known_speaker_id", "unknown"),
            ),
        ),
        expected_artifacts=(
            ArtifactContract(
                artifact_id="task_contract_adr",
                required_for_handoff=True,
                path_hint="docs/model-task-contract.md",
                producer="checked-in repository documentation",
                description=(
                    "Human-readable ADR that fixes the task formulation, trial types, artifact "
                    "expectations, and scope limits."
                ),
            ),
            ArtifactContract(
                artifact_id="task_contract_snapshot_json",
                required_for_handoff=True,
                path_hint=(f"{output_root}/{MODEL_TASK_CONTRACT_JSON_NAME}"),
                producer="scripts/build_model_task_contract.py",
                description=(
                    "Machine-readable snapshot of the current repository contract for downstream "
                    "automation and release handoff."
                ),
            ),
            ArtifactContract(
                artifact_id="task_contract_snapshot_markdown",
                required_for_handoff=True,
                path_hint=(f"{output_root}/{MODEL_TASK_CONTRACT_MARKDOWN_NAME}"),
                producer="scripts/build_model_task_contract.py",
                description=("Generated markdown companion to the JSON snapshot for local review."),
            ),
            ArtifactContract(
                artifact_id="verification_report_bundle",
                required_for_handoff=True,
                path_hint="artifacts/**/verification_eval_report.json",
                producer="training pipelines + scripts/evaluate_verification_scores.py",
                description=(
                    "Offline verification quality bundle that proves the primary task on dev "
                    "trials."
                ),
            ),
            ArtifactContract(
                artifact_id="threshold_bundle",
                required_for_handoff=True,
                path_hint="artifacts/**/verification_threshold_calibration.json",
                producer="scripts/calibrate_verification_thresholds.py",
                description=(
                    "Named decision thresholds that turn cosine scores into demo or "
                    "production-like verification decisions."
                ),
            ),
            ArtifactContract(
                artifact_id="export_boundary_contract",
                required_for_handoff=True,
                path_hint="artifacts/export-boundary/export_boundary.json",
                producer="scripts/export_boundary_report.py",
                description=(
                    "Encoder input/output boundary that keeps the task contract aligned with "
                    "future ONNX/TensorRT export work."
                ),
            ),
        ),
        limitations=(
            ContractLimitation(
                limitation_id="runtime_backend_is_demo_grade",
                impact="runtime_quality",
                description=(
                    "The checked-in runtime still uses the `feature_statistics` inferencer, so a "
                    "healthy demo/runtime path validates contract shape more than final learned "
                    "speaker quality."
                ),
            ),
            ContractLimitation(
                limitation_id="identification_modes_not_productized",
                impact="task_surface",
                description=(
                    "Closed-set and open-set identification remain compatibility targets over the "
                    "shared embedding space, not first-class report builders or serving endpoints."
                ),
            ),
            ContractLimitation(
                limitation_id="thresholds_are_candidate_specific",
                impact="decision_semantics",
                description=(
                    "Verification decisions are only valid together with the threshold bundle "
                    "frozen for the active score distribution and candidate model."
                ),
            ),
            ContractLimitation(
                limitation_id="enrollment_cache_is_bundle_specific",
                impact="runtime_state",
                description=(
                    "Enrollment caches and runtime enrollment state are only valid for the "
                    "matching model-bundle metadata and export boundary contract."
                ),
            ),
        ),
        supporting_docs=(
            "docs/embedding-scoring.md",
            "docs/evaluation-package.md",
            "docs/export-boundary.md",
            "docs/model-card.md",
            "docs/unified-inference-wrapper.md",
        ),
    )


def render_model_task_contract_markdown(contract: ModelTaskContract) -> str:
    lines = [
        f"# {contract.title}",
        "",
        f"`{contract.ticket_id}` fixes one explicit repository contract for the Kryptonite speaker "
        "stack.",
        "",
        "## Decision",
        "",
        f"- primary task mode: `{contract.primary_task_mode}`",
        f"- current runtime backend: `{contract.embedding_contract.inferencer_backend}`",
        "- compatibility modes: `closed_set_identification`, `open_set_identification`",
        "",
        contract.summary,
        "",
        "## Canonical Workflow",
        "",
    ]
    for index, step in enumerate(contract.canonical_workflow, start=1):
        lines.append(f"{index}. {step}")

    lines.extend(
        [
            "",
            "## Input And Output Contract",
            "",
            f"- raw audio target: `{contract.raw_audio_contract.sample_rate_hz} Hz`, "
            f"`{contract.raw_audio_contract.channels}` channel(s), "
            f"`{contract.raw_audio_contract.output_format}`/"
            f"`PCM{contract.raw_audio_contract.pcm_bits_per_sample}`",
            f"- loudness mode: `{contract.raw_audio_contract.loudness_mode}`",
            f"- VAD mode: `{contract.raw_audio_contract.vad_mode}`",
            f"- train crop window: `{contract.raw_audio_contract.train_crop_seconds[0]}-"
            f"{contract.raw_audio_contract.train_crop_seconds[1]} s`",
            f"- eval chunk / overlap: `{contract.raw_audio_contract.eval_chunk_seconds} s` / "
            f"`{contract.raw_audio_contract.eval_chunk_overlap_seconds} s`",
            f"- demo chunk / overlap: `{contract.raw_audio_contract.demo_chunk_seconds} s` / "
            f"`{contract.raw_audio_contract.demo_chunk_overlap_seconds} s`",
            f"- encoder boundary: `{contract.embedding_contract.boundary_mode}`",
            f"- encoder input: `{contract.embedding_contract.input_signature}`",
            f"- encoder output: `{contract.embedding_contract.output_signature}`",
            f"- scoring metric: `{contract.embedding_contract.scoring_metric}`",
            f"- enrollment pooling: `{', '.join(contract.embedding_contract.enrollment_pooling)}`",
            "",
            "## Task Modes",
            "",
        ]
    )
    for mode in contract.task_modes:
        lines.extend(
            [
                f"### {mode.mode_id}",
                "",
                f"- support level: `{mode.support_level}`",
                f"- request unit: `{mode.request_unit}`",
                f"- score output: {mode.score_output}",
                f"- decision output: {mode.decision_output}",
                f"- metric focus: `{', '.join(mode.metric_focus)}`",
                f"- summary: {mode.summary}",
            ]
        )
        if mode.notes:
            lines.append(f"- notes: {' '.join(mode.notes)}")
        lines.append("")

    lines.extend(["## Trial Types", ""])
    for trial in contract.trial_types:
        lines.extend(
            [
                f"### {trial.trial_type_id}",
                "",
                f"- mode: `{trial.mode_id}`",
                f"- support level: `{trial.support_level}`",
                f"- evaluation unit: {trial.unit_of_evaluation}",
                f"- required fields: `{', '.join(trial.required_fields)}`",
                f"- label space: `{', '.join(trial.label_space)}`",
                f"- description: {trial.description}",
            ]
        )
        if trial.slice_fields:
            lines.append(f"- slice fields: `{', '.join(trial.slice_fields)}`")
        if trial.source_path is not None:
            lines.append(f"- source path: `{trial.source_path}`")
        lines.append("")

    lines.extend(["## Expected Artifacts", ""])
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
        lines.extend(
            [
                f"- `{limitation.limitation_id}` ({limitation.impact}): {limitation.description}",
            ]
        )

    lines.extend(
        [
            "",
            "## Supporting References",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in contract.supporting_docs)
    lines.extend(
        [
            "",
            "## Rebuild",
            "",
            "```bash",
            "uv run python scripts/build_model_task_contract.py --config configs/base.toml",
            "```",
            "",
            f"This writes `{contract.output_root}/{MODEL_TASK_CONTRACT_JSON_NAME}` and "
            f"`{contract.output_root}/{MODEL_TASK_CONTRACT_MARKDOWN_NAME}`.",
        ]
    )
    return "\n".join(lines)


def write_model_task_contract(
    contract: ModelTaskContract,
    *,
    project_root: Path | str = ".",
) -> WrittenModelTaskContract:
    root = Path(project_root)
    output_root = root / contract.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    report_json_path = output_root / MODEL_TASK_CONTRACT_JSON_NAME
    report_markdown_path = output_root / MODEL_TASK_CONTRACT_MARKDOWN_NAME
    report_json_path.write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_markdown_path.write_text(
        render_model_task_contract_markdown(contract) + "\n",
        encoding="utf-8",
    )
    return WrittenModelTaskContract(
        output_root=str(output_root),
        report_json_path=str(report_json_path),
        report_markdown_path=str(report_markdown_path),
    )


def _render_tensor_signature(tensor: Any) -> str:
    axes = ", ".join(f"{axis.name}={axis.size}" for axis in tensor.axes)
    return f"{tensor.name} ({tensor.layout}, {tensor.dtype}; {axes})"


__all__ = [
    "DEFAULT_MODEL_TASK_CONTRACT_OUTPUT_ROOT",
    "MODEL_TASK_CONTRACT_FORMAT_VERSION",
    "MODEL_TASK_CONTRACT_JSON_NAME",
    "MODEL_TASK_CONTRACT_MARKDOWN_NAME",
    "ArtifactContract",
    "ContractLimitation",
    "EmbeddingContract",
    "ModelTaskContract",
    "RawAudioContract",
    "TaskModeContract",
    "TrialTypeContract",
    "WrittenModelTaskContract",
    "build_model_task_contract",
    "render_model_task_contract_markdown",
    "write_model_task_contract",
]
