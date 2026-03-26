"""Select the final CAM++ stage-3 candidate and evaluate optional checkpoint averages."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from kryptonite.data import AudioLoadRequest
from kryptonite.deployment import resolve_project_path
from kryptonite.eval import (
    build_verification_evaluation_report,
    load_verification_score_rows,
    load_verification_trial_rows,
    write_verification_evaluation_report,
)
from kryptonite.features import FbankExtractionRequest
from kryptonite.models import CAMPPlusEncoder
from kryptonite.tracking import create_run_id

from ..manifest_speaker_data import load_manifest_rows
from ..speaker_baseline import (
    SCORE_SUMMARY_FILE_NAME,
    export_dev_embeddings,
    relative_to_project,
    resolve_device,
    score_trials,
)
from .model_selection_config import CAMPPlusModelSelectionConfig
from .stage3_config import CAMPPlusStage3Config, load_campp_stage3_config
from .sweep_shortlist import (
    SweepCandidateResult,
    SweepShortlistRunArtifacts,
    SweepShortlistSummary,
    SweepSuiteEvaluation,
    load_campp_sweep_shortlist_report,
)

MODEL_SELECTION_REPORT_JSON_NAME = "campp_model_selection_report.json"
MODEL_SELECTION_REPORT_MARKDOWN_NAME = "campp_model_selection_report.md"
VARIANT_CHECKPOINT_NAME = "campp_stage3_encoder.pt"
VARIANT_REPORT_NAME = "variant_report.md"
FINAL_CANDIDATE_DIR_NAME = "final_candidate"
FINAL_CANDIDATE_METADATA_NAME = "final_candidate_selection.json"


@dataclass(frozen=True, slots=True)
class ModelSelectionVariant:
    """One evaluated final-candidate option."""

    variant_id: str
    description: str
    rank: int
    uses_checkpoint_averaging: bool
    source_candidate_ids: tuple[str, ...]
    source_checkpoint_paths: tuple[str, ...]
    checkpoint_path: str
    output_root: str
    report_markdown_path: str
    selection_score: float
    weighted_eer: float
    weighted_min_dcf: float
    clean_eer: float | None
    clean_min_dcf: float | None
    robust_eer: float
    robust_min_dcf: float
    final_train_loss: float
    final_train_accuracy: float
    suites: tuple[SweepSuiteEvaluation, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source_candidate_ids"] = list(self.source_candidate_ids)
        payload["source_checkpoint_paths"] = list(self.source_checkpoint_paths)
        payload["suites"] = [suite.to_dict() for suite in self.suites]
        return payload


@dataclass(frozen=True, slots=True)
class ModelSelectionSummary:
    """Top-level metadata for the final-candidate selection run."""

    generated_at: str
    config_path: str
    shortlist_report_path: str
    shortlist_winner_candidate_id: str
    shortlist_winner_selection_score: float
    base_stage3_config_path: str
    output_root: str
    clean_weight: float
    corrupted_weight: float
    eer_weight: float
    min_dcf_weight: float
    corrupted_suite_ids: tuple[str, ...]
    evaluated_variant_count: int
    skipped_variants: tuple[str, ...]
    selected_variant_id: str
    selected_checkpoint_path: str
    final_candidate_dir: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["corrupted_suite_ids"] = list(self.corrupted_suite_ids)
        payload["skipped_variants"] = list(self.skipped_variants)
        return payload


@dataclass(frozen=True, slots=True)
class ModelSelectionArtifacts:
    """Written outputs for one model-selection run."""

    output_root: str
    report_json_path: str
    report_markdown_path: str
    final_candidate_dir: str
    final_checkpoint_path: str
    summary: ModelSelectionSummary
    variants: tuple[ModelSelectionVariant, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "final_candidate_dir": self.final_candidate_dir,
            "final_checkpoint_path": self.final_checkpoint_path,
            "summary": self.summary.to_dict(),
            "variants": [variant.to_dict() for variant in self.variants],
        }


@dataclass(frozen=True, slots=True)
class _WeightedMetrics:
    selection_score: float
    weighted_eer: float
    weighted_min_dcf: float
    clean_eer: float | None
    clean_min_dcf: float | None
    robust_eer: float
    robust_min_dcf: float


def run_campp_model_selection(
    config: CAMPPlusModelSelectionConfig,
    *,
    config_path: Path | str,
    env_file: Path | str | None = None,
    device_override: str | None = None,
) -> ModelSelectionArtifacts:
    """Evaluate the shortlist winner and optional averaged variants."""

    shortlist_report_path = resolve_project_path(".", config.shortlist_report_path)
    shortlist = load_campp_sweep_shortlist_report(report_path=shortlist_report_path)
    project_root = _infer_project_root(
        shortlist_report_path=shortlist_report_path,
        shortlist_output_root=shortlist.output_root,
    )
    winner = shortlist.candidates[0]
    winner_config = _load_winner_stage3_config(
        shortlist=shortlist,
        winner=winner,
        env_file=env_file,
        project_root=project_root,
    )

    output_root = resolve_project_path(str(project_root), config.output_root) / create_run_id()
    output_root.mkdir(parents=True, exist_ok=True)

    variants: list[ModelSelectionVariant] = [
        _build_raw_winner_variant(winner=winner),
    ]
    skipped_variants: list[str] = []
    if config.averaging.enabled:
        for candidate_count in config.averaging.candidate_counts:
            if len(shortlist.candidates) < candidate_count:
                skipped_variants.append(
                    "top"
                    f"{candidate_count}_uniform_average: shortlist has only "
                    f"{len(shortlist.candidates)} candidates."
                )
                continue
            candidate_group = shortlist.candidates[:candidate_count]
            compatibility_issue = _check_candidate_group_compatibility(
                candidate_group=candidate_group,
                project_root=project_root,
            )
            if compatibility_issue is not None:
                skipped_variants.append(
                    f"top{candidate_count}_uniform_average: {compatibility_issue}"
                )
                continue
            variants.append(
                _build_averaged_variant(
                    variant_id=f"top{candidate_count}_uniform_average",
                    candidate_group=candidate_group,
                    winner=winner,
                    winner_config=winner_config,
                    shortlist_summary=shortlist.summary,
                    project_root=project_root,
                    output_root=output_root,
                    device_override=device_override,
                )
            )

    ranked_variants = _rank_variants(variants)
    selected_variant = ranked_variants[0]
    final_candidate_dir = output_root / FINAL_CANDIDATE_DIR_NAME
    final_candidate_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint_path = final_candidate_dir / VARIANT_CHECKPOINT_NAME
    shutil.copy2(
        resolve_project_path(str(project_root), selected_variant.checkpoint_path),
        final_checkpoint_path,
    )
    final_metadata_path = final_candidate_dir / FINAL_CANDIDATE_METADATA_NAME
    final_metadata_path.write_text(
        json.dumps(
            {
                "selected_variant_id": selected_variant.variant_id,
                "source_candidate_ids": list(selected_variant.source_candidate_ids),
                "selection_score": selected_variant.selection_score,
                "checkpoint_path": selected_variant.checkpoint_path,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = ModelSelectionSummary(
        generated_at=_utc_now(),
        config_path=relative_to_project(Path(config_path), project_root=project_root),
        shortlist_report_path=relative_to_project(shortlist_report_path, project_root=project_root),
        shortlist_winner_candidate_id=winner.candidate_id,
        shortlist_winner_selection_score=winner.selection_score,
        base_stage3_config_path=shortlist.summary.base_stage3_config_path,
        output_root=relative_to_project(output_root, project_root=project_root),
        clean_weight=shortlist.summary.clean_weight,
        corrupted_weight=shortlist.summary.corrupted_weight,
        eer_weight=shortlist.summary.eer_weight,
        min_dcf_weight=shortlist.summary.min_dcf_weight,
        corrupted_suite_ids=shortlist.summary.corrupted_suite_ids,
        evaluated_variant_count=len(ranked_variants),
        skipped_variants=tuple(skipped_variants),
        selected_variant_id=selected_variant.variant_id,
        selected_checkpoint_path=relative_to_project(
            final_checkpoint_path,
            project_root=project_root,
        ),
        final_candidate_dir=relative_to_project(final_candidate_dir, project_root=project_root),
    )
    report_json_path = output_root / MODEL_SELECTION_REPORT_JSON_NAME
    report_markdown_path = output_root / MODEL_SELECTION_REPORT_MARKDOWN_NAME
    artifacts = ModelSelectionArtifacts(
        output_root=summary.output_root,
        report_json_path=relative_to_project(report_json_path, project_root=project_root),
        report_markdown_path=relative_to_project(report_markdown_path, project_root=project_root),
        final_candidate_dir=summary.final_candidate_dir,
        final_checkpoint_path=summary.selected_checkpoint_path,
        summary=summary,
        variants=tuple(ranked_variants),
    )
    report_json_path.write_text(json.dumps(artifacts.to_dict(), indent=2, sort_keys=True) + "\n")
    report_markdown_path.write_text(
        render_campp_model_selection_markdown(artifacts) + "\n",
        encoding="utf-8",
    )
    return artifacts


def render_campp_model_selection_markdown(artifacts: ModelSelectionArtifacts) -> str:
    """Render the final-candidate selector outputs as markdown."""

    summary = artifacts.summary
    selected = artifacts.variants[0]
    lines = [
        "# CAM++ Stage-3 Model Selection",
        "",
        f"- Output root: `{summary.output_root}`",
        f"- Config: `{summary.config_path}`",
        f"- Shortlist report: `{summary.shortlist_report_path}`",
        f"- Shortlist winner: `{summary.shortlist_winner_candidate_id}`",
        (
            "- Ranking objective: "
            f"`{summary.eer_weight:.2f} * weighted_eer + "
            f"{summary.min_dcf_weight:.2f} * weighted_min_dcf`, "
            f"with clean/corrupted weights `{summary.clean_weight:.2f}` / "
            f"`{summary.corrupted_weight:.2f}`"
        ),
        f"- Final candidate dir: `{summary.final_candidate_dir}`",
        f"- Final checkpoint: `{summary.selected_checkpoint_path}`",
        "",
        "## Winner",
        "",
        f"- Variant: `{selected.variant_id}`",
        f"- Description: {selected.description}",
        f"- Averaged: `{selected.uses_checkpoint_averaging}`",
        f"- Source candidates: `{list(selected.source_candidate_ids)}`",
        f"- Selection score: `{selected.selection_score:.6f}`",
        (
            f"- Weighted EER / minDCF: `{selected.weighted_eer:.6f}` / "
            f"`{selected.weighted_min_dcf:.6f}`"
        ),
        f"- Robust EER / minDCF: `{selected.robust_eer:.6f}` / `{selected.robust_min_dcf:.6f}`",
        f"- Checkpoint: `{selected.checkpoint_path}`",
        "",
        "## Variants",
        "",
        (
            "| Rank | Variant | Averaged | Score | Weighted EER | Weighted minDCF | "
            "Robust EER | Clean EER | Sources |"
        ),
        "|------|---------|----------|-------|--------------|-----------------|------------|-----------|---------|",
    ]
    for variant in artifacts.variants:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(variant.rank),
                    f"`{variant.variant_id}`",
                    str(variant.uses_checkpoint_averaging),
                    f"{variant.selection_score:.6f}",
                    f"{variant.weighted_eer:.6f}",
                    f"{variant.weighted_min_dcf:.6f}",
                    f"{variant.robust_eer:.6f}",
                    ("-" if variant.clean_eer is None else f"{variant.clean_eer:.6f}"),
                    f"`{list(variant.source_candidate_ids)}`",
                ]
            )
            + " |"
        )
    if summary.skipped_variants:
        lines.extend(["", "## Skipped Variants", ""])
        for item in summary.skipped_variants:
            lines.append(f"- {item}")
    lines.extend(["", "## Suite Breakdown", ""])
    for variant in artifacts.variants:
        lines.extend(
            [
                f"### `{variant.variant_id}`",
                "",
                f"- Description: {variant.description}",
                f"- Output root: `{variant.output_root}`",
                f"- Report: `{variant.report_markdown_path}`",
                "",
                "| Suite | Family | EER | minDCF | Trials | Report |",
                "|------|--------|-----|--------|--------|--------|",
            ]
        )
        for suite in variant.suites:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{suite.label}`",
                        suite.family,
                        f"{suite.eer:.6f}",
                        f"{suite.min_dcf:.6f}",
                        str(suite.trial_count),
                        f"`{suite.report_markdown_path}`",
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def _load_winner_stage3_config(
    *,
    shortlist: SweepShortlistRunArtifacts,
    winner: SweepCandidateResult,
    env_file: Path | str | None,
    project_root: Path,
) -> CAMPPlusStage3Config:
    base_stage3_config_path = resolve_project_path(
        str(project_root),
        shortlist.summary.base_stage3_config_path,
    )
    return load_campp_stage3_config(
        config_path=base_stage3_config_path,
        env_file=env_file,
        project_overrides=list(winner.project_overrides),
    )


def _build_raw_winner_variant(*, winner: SweepCandidateResult) -> ModelSelectionVariant:
    return ModelSelectionVariant(
        variant_id="winner_raw",
        description="Direct shortlist winner without additional checkpoint averaging.",
        rank=0,
        uses_checkpoint_averaging=False,
        source_candidate_ids=(winner.candidate_id,),
        source_checkpoint_paths=(winner.checkpoint_path,),
        checkpoint_path=winner.checkpoint_path,
        output_root=winner.run_output_root,
        report_markdown_path=winner.run_report_path,
        selection_score=winner.selection_score,
        weighted_eer=winner.weighted_eer,
        weighted_min_dcf=winner.weighted_min_dcf,
        clean_eer=winner.clean_eer,
        clean_min_dcf=winner.clean_min_dcf,
        robust_eer=winner.robust_eer,
        robust_min_dcf=winner.robust_min_dcf,
        final_train_loss=winner.final_train_loss,
        final_train_accuracy=winner.final_train_accuracy,
        suites=winner.suites,
    )


def _check_candidate_group_compatibility(
    *,
    candidate_group: tuple[SweepCandidateResult, ...],
    project_root: Path,
) -> str | None:
    reference = _load_checkpoint_payload(
        checkpoint_path=resolve_project_path(str(project_root), candidate_group[0].checkpoint_path)
    )
    reference_model_config = reference.get("model_config")
    reference_speaker_map = reference.get("speaker_to_index")
    if not isinstance(reference_model_config, dict) or not isinstance(reference_speaker_map, dict):
        return "reference checkpoint is missing model_config or speaker_to_index."
    reference_model_state = reference.get("model_state_dict")
    reference_classifier_state = reference.get("classifier_state_dict")
    if not isinstance(reference_model_state, dict) or not isinstance(
        reference_classifier_state, dict
    ):
        return "reference checkpoint is missing model/classifier state."
    for candidate in candidate_group[1:]:
        payload = _load_checkpoint_payload(
            checkpoint_path=resolve_project_path(str(project_root), candidate.checkpoint_path)
        )
        if payload.get("model_config") != reference_model_config:
            return f"candidate {candidate.candidate_id} has a different model_config."
        if payload.get("speaker_to_index") != reference_speaker_map:
            return f"candidate {candidate.candidate_id} has a different speaker_to_index mapping."
        model_state = payload.get("model_state_dict")
        classifier_state = payload.get("classifier_state_dict")
        if not isinstance(model_state, dict) or not isinstance(classifier_state, dict):
            return f"candidate {candidate.candidate_id} is missing model/classifier state."
        if _state_dict_signature(model_state) != _state_dict_signature(reference_model_state):
            return f"candidate {candidate.candidate_id} has incompatible model tensor shapes."
        if _state_dict_signature(classifier_state) != _state_dict_signature(
            reference_classifier_state
        ):
            return f"candidate {candidate.candidate_id} has incompatible classifier tensor shapes."
    return None


def _build_averaged_variant(
    *,
    variant_id: str,
    candidate_group: tuple[SweepCandidateResult, ...],
    winner: SweepCandidateResult,
    winner_config: CAMPPlusStage3Config,
    shortlist_summary: SweepShortlistSummary,
    project_root: Path,
    output_root: Path,
    device_override: str | None,
) -> ModelSelectionVariant:
    source_paths = tuple(
        resolve_project_path(str(project_root), candidate.checkpoint_path)
        for candidate in candidate_group
    )
    payloads = [_load_checkpoint_payload(checkpoint_path=path) for path in source_paths]
    variant_output_root = output_root / "variants" / variant_id
    variant_output_root.mkdir(parents=True, exist_ok=True)
    averaged_checkpoint_path = variant_output_root / VARIANT_CHECKPOINT_NAME
    averaged_payload = _average_checkpoint_payloads(
        payloads=payloads,
        source_candidate_ids=tuple(candidate.candidate_id for candidate in candidate_group),
    )
    torch.save(averaged_payload, averaged_checkpoint_path)

    suite_results = _evaluate_variant_suites(
        checkpoint_path=averaged_checkpoint_path,
        winner=winner,
        winner_config=winner_config,
        project_root=project_root,
        output_root=variant_output_root,
        device_override=device_override,
    )
    weighted_metrics = _compute_weighted_metrics(
        suites=suite_results,
        summary=shortlist_summary,
    )
    report_path = variant_output_root / VARIANT_REPORT_NAME
    source_checkpoint_paths = tuple(
        relative_to_project(path, project_root=project_root) for path in source_paths
    )
    report_path.write_text(
        _render_variant_markdown(
            variant_id=variant_id,
            source_candidate_ids=tuple(candidate.candidate_id for candidate in candidate_group),
            source_checkpoint_paths=source_checkpoint_paths,
            suite_results=suite_results,
            weighted_metrics=weighted_metrics,
            output_root=variant_output_root,
            project_root=project_root,
        )
        + "\n",
        encoding="utf-8",
    )
    mean_train_loss = round(
        sum(candidate.final_train_loss for candidate in candidate_group) / len(candidate_group),
        6,
    )
    mean_train_accuracy = round(
        sum(candidate.final_train_accuracy for candidate in candidate_group) / len(candidate_group),
        6,
    )
    return ModelSelectionVariant(
        variant_id=variant_id,
        description=(
            f"Uniform checkpoint average over the top {len(candidate_group)} shortlist candidates."
        ),
        rank=0,
        uses_checkpoint_averaging=True,
        source_candidate_ids=tuple(candidate.candidate_id for candidate in candidate_group),
        source_checkpoint_paths=source_checkpoint_paths,
        checkpoint_path=relative_to_project(averaged_checkpoint_path, project_root=project_root),
        output_root=relative_to_project(variant_output_root, project_root=project_root),
        report_markdown_path=relative_to_project(report_path, project_root=project_root),
        selection_score=weighted_metrics.selection_score,
        weighted_eer=weighted_metrics.weighted_eer,
        weighted_min_dcf=weighted_metrics.weighted_min_dcf,
        clean_eer=weighted_metrics.clean_eer,
        clean_min_dcf=weighted_metrics.clean_min_dcf,
        robust_eer=weighted_metrics.robust_eer,
        robust_min_dcf=weighted_metrics.robust_min_dcf,
        final_train_loss=mean_train_loss,
        final_train_accuracy=mean_train_accuracy,
        suites=suite_results,
    )


def _evaluate_variant_suites(
    *,
    checkpoint_path: Path,
    winner: SweepCandidateResult,
    winner_config: CAMPPlusStage3Config,
    project_root: Path,
    output_root: Path,
    device_override: str | None,
) -> tuple[SweepSuiteEvaluation, ...]:
    evaluation_device = resolve_device(device_override or winner_config.project.runtime.device)
    model = CAMPPlusEncoder(winner_config.model).to(evaluation_device)
    payload = _load_checkpoint_payload(checkpoint_path=checkpoint_path)
    model_state = payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError(f"Averaged checkpoint {checkpoint_path} is missing model_state_dict.")
    model.load_state_dict(model_state)
    model.eval()

    audio_request = AudioLoadRequest.from_config(
        winner_config.project.normalization,
        vad=winner_config.project.vad,
    )
    feature_request = FbankExtractionRequest.from_config(winner_config.project.features)
    suite_results: list[SweepSuiteEvaluation] = []
    for suite in winner.suites:
        suite_output_root = output_root / suite.suite_id
        suite_output_root.mkdir(parents=True, exist_ok=True)
        suite_rows = load_manifest_rows(
            suite.manifest_path,
            project_root=project_root,
            limit=winner_config.data.max_dev_rows,
        )
        embedding_summary, metadata_rows = export_dev_embeddings(
            output_root=suite_output_root,
            model=model,
            rows=suite_rows,
            manifest_path=suite.manifest_path,
            project_root=project_root,
            audio_request=audio_request,
            feature_request=feature_request,
            chunking=winner_config.project.chunking,
            device=evaluation_device,
            embedding_source=f"campp_stage3_model_selection:{suite.suite_id}",
        )
        trials_path = resolve_project_path(str(project_root), suite.trials_path)
        trial_rows = [dict(row) for row in load_verification_trial_rows(trials_path)]
        score_summary = score_trials(
            output_root=suite_output_root,
            trials_path=trials_path,
            metadata_rows=metadata_rows,
            trial_rows=trial_rows,
        )
        score_summary_path = suite_output_root / SCORE_SUMMARY_FILE_NAME
        score_summary_path.write_text(
            json.dumps(score_summary.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verification_report = write_verification_evaluation_report(
            build_verification_evaluation_report(
                load_verification_score_rows(score_summary.scores_path),
                scores_path=score_summary.scores_path,
                trials_path=trials_path,
                metadata_path=embedding_summary.metadata_parquet_path,
                trial_rows=trial_rows,
                metadata_rows=metadata_rows,
            ),
            output_root=suite_output_root,
        )
        metrics = verification_report.summary.metrics
        suite_results.append(
            SweepSuiteEvaluation(
                suite_id=suite.suite_id,
                label=suite.label,
                family=suite.family,
                manifest_path=suite.manifest_path,
                trials_path=relative_to_project(trials_path, project_root=project_root),
                output_root=relative_to_project(suite_output_root, project_root=project_root),
                report_markdown_path=relative_to_project(
                    Path(verification_report.report_markdown_path),
                    project_root=project_root,
                ),
                trial_count=metrics.trial_count,
                eer=metrics.eer,
                min_dcf=metrics.min_dcf,
                score_gap=score_summary.score_gap,
            )
        )
    return tuple(suite_results)


def _average_checkpoint_payloads(
    *,
    payloads: list[dict[str, Any]],
    source_candidate_ids: tuple[str, ...],
) -> dict[str, Any]:
    model_states = [
        _require_tensor_state(payload.get("model_state_dict"), "model_state_dict")
        for payload in payloads
    ]
    classifier_states = [
        _require_tensor_state(payload.get("classifier_state_dict"), "classifier_state_dict")
        for payload in payloads
    ]
    reference_payload = payloads[0]
    baseline_config = reference_payload.get("baseline_config")
    baseline_config_dict = dict(baseline_config) if isinstance(baseline_config, dict) else {}
    baseline_config_dict["selection"] = {
        "strategy": "uniform_checkpoint_average",
        "source_candidate_ids": list(source_candidate_ids),
        "source_checkpoint_count": len(payloads),
    }
    model_config = reference_payload.get("model_config")
    speaker_to_index = reference_payload.get("speaker_to_index")
    return {
        "model_state_dict": _average_state_dicts(model_states),
        "classifier_state_dict": _average_state_dicts(classifier_states),
        "model_config": ({} if not isinstance(model_config, dict) else dict(model_config)),
        "baseline_config": baseline_config_dict,
        "speaker_to_index": (
            {} if not isinstance(speaker_to_index, dict) else dict(speaker_to_index)
        ),
    }


def _average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    averaged: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        tensors = [state_dict[key] for state_dict in state_dicts]
        reference = tensors[0]
        if torch.is_floating_point(reference):
            stacked = torch.stack([tensor.to(dtype=torch.float64) for tensor in tensors], dim=0)
            averaged[key] = stacked.mean(dim=0).to(dtype=reference.dtype)
            continue
        averaged[key] = reference.clone()
    return averaged


def _require_tensor_state(payload: object, field_name: str) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be a state-dict mapping.")
    state = dict(payload)
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"{field_name} must contain string -> Tensor entries.")
    return state


def _load_checkpoint_payload(*, checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain an object payload.")
    return payload


def _state_dict_signature(
    state_dict: dict[str, torch.Tensor],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(sorted((key, tuple(value.shape)) for key, value in state_dict.items()))


def _compute_weighted_metrics(
    *,
    suites: tuple[SweepSuiteEvaluation, ...],
    summary: SweepShortlistSummary,
) -> _WeightedMetrics:
    clean_suite = next((suite for suite in suites if suite.suite_id == "clean_dev"), None)
    robust_suites = tuple(suite for suite in suites if suite.suite_id != "clean_dev")
    if not robust_suites:
        raise ValueError("Model selection requires at least one corrupted suite.")
    robust_eer = round(sum(suite.eer for suite in robust_suites) / len(robust_suites), 6)
    robust_min_dcf = round(sum(suite.min_dcf for suite in robust_suites) / len(robust_suites), 6)
    if clean_suite is None:
        normalized_clean_weight = 0.0
        normalized_corrupted_weight = 1.0
    else:
        total = summary.clean_weight + summary.corrupted_weight
        normalized_clean_weight = summary.clean_weight / total
        normalized_corrupted_weight = summary.corrupted_weight / total
    weighted_eer = round(
        normalized_clean_weight * (robust_eer if clean_suite is None else clean_suite.eer)
        + normalized_corrupted_weight * robust_eer,
        6,
    )
    weighted_min_dcf = round(
        normalized_clean_weight * (robust_min_dcf if clean_suite is None else clean_suite.min_dcf)
        + normalized_corrupted_weight * robust_min_dcf,
        6,
    )
    selection_score = round(
        summary.eer_weight * weighted_eer + summary.min_dcf_weight * weighted_min_dcf,
        6,
    )
    return _WeightedMetrics(
        selection_score=selection_score,
        weighted_eer=weighted_eer,
        weighted_min_dcf=weighted_min_dcf,
        clean_eer=(None if clean_suite is None else clean_suite.eer),
        clean_min_dcf=(None if clean_suite is None else clean_suite.min_dcf),
        robust_eer=robust_eer,
        robust_min_dcf=robust_min_dcf,
    )


def _render_variant_markdown(
    *,
    variant_id: str,
    source_candidate_ids: tuple[str, ...],
    source_checkpoint_paths: tuple[str, ...],
    suite_results: tuple[SweepSuiteEvaluation, ...],
    weighted_metrics: _WeightedMetrics,
    output_root: Path,
    project_root: Path,
) -> str:
    lines = [
        f"# {variant_id}",
        "",
        f"- Output root: `{relative_to_project(output_root, project_root=project_root)}`",
        f"- Source candidates: `{list(source_candidate_ids)}`",
        f"- Source checkpoints: `{list(source_checkpoint_paths)}`",
        f"- Selection score: `{weighted_metrics.selection_score}`",
        (
            f"- Weighted EER / minDCF: `{weighted_metrics.weighted_eer}` / "
            f"`{weighted_metrics.weighted_min_dcf}`"
        ),
        "",
        "| Suite | Family | EER | minDCF | Trials | Report |",
        "|------|--------|-----|--------|--------|--------|",
    ]
    for suite in suite_results:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{suite.label}`",
                    suite.family,
                    f"{suite.eer:.6f}",
                    f"{suite.min_dcf:.6f}",
                    str(suite.trial_count),
                    f"`{suite.report_markdown_path}`",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _rank_variants(variants: list[ModelSelectionVariant]) -> list[ModelSelectionVariant]:
    ordered = sorted(
        variants,
        key=lambda variant: (
            variant.selection_score,
            variant.weighted_eer,
            variant.weighted_min_dcf,
            variant.variant_id,
        ),
    )
    return [replace(variant, rank=index + 1) for index, variant in enumerate(ordered)]


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _infer_project_root(
    *,
    shortlist_report_path: Path,
    shortlist_output_root: str,
) -> Path:
    output_root_path = Path(shortlist_output_root)
    if output_root_path.is_absolute():
        return output_root_path.parent
    absolute_output_root = shortlist_report_path.resolve().parent
    if not output_root_path.parts:
        return absolute_output_root
    return absolute_output_root.parents[len(output_root_path.parts) - 1]


__all__ = [
    "FINAL_CANDIDATE_DIR_NAME",
    "MODEL_SELECTION_REPORT_JSON_NAME",
    "MODEL_SELECTION_REPORT_MARKDOWN_NAME",
    "ModelSelectionArtifacts",
    "ModelSelectionSummary",
    "ModelSelectionVariant",
    "render_campp_model_selection_markdown",
    "run_campp_model_selection",
]
