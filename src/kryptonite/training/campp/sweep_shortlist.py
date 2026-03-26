"""Run a bounded CAM++ stage-3 shortlist and rank candidates on robust dev suites."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

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
    build_default_cohort_bank,
    export_dev_embeddings,
    load_or_generate_trials,
    relative_to_project,
    resolve_device,
    score_trials,
)
from .stage3_config import CAMPPlusStage3Config, Stage3Config
from .stage3_pipeline import run_campp_stage3
from .sweep_shortlist_config import (
    CAMPPlusSweepShortlistConfig,
    SweepCandidateConfig,
    SweepCropCurriculumOverride,
    SweepMarginScheduleOverride,
    SweepSelectionConfig,
)

SHORTLIST_REPORT_JSON_NAME = "campp_stage3_sweep_shortlist_report.json"
SHORTLIST_REPORT_MARKDOWN_NAME = "campp_stage3_sweep_shortlist_report.md"
SUITE_TRIALS_FILE_NAME = "suite_trials.jsonl"


@dataclass(frozen=True, slots=True)
class SweepSuiteEvaluation:
    """Metrics for one clean/corrupted dev evaluation suite."""

    suite_id: str
    label: str
    family: str
    manifest_path: str
    trials_path: str
    output_root: str
    report_markdown_path: str
    trial_count: int
    eer: float
    min_dcf: float
    score_gap: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SweepCandidateResult:
    """Resolved metrics and config summary for one shortlist candidate."""

    candidate_id: str
    description: str
    rank: int
    selection_score: float
    weighted_eer: float
    weighted_min_dcf: float
    clean_eer: float | None
    clean_min_dcf: float | None
    robust_eer: float
    robust_min_dcf: float
    train_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    eval_pooling: str
    crop_start_seconds: float
    crop_end_seconds: float
    margin_start: float
    margin_end: float
    max_epochs: int
    final_train_loss: float
    final_train_accuracy: float
    checkpoint_path: str
    run_output_root: str
    run_report_path: str
    tracking_run_dir: str | None
    project_overrides: tuple[str, ...]
    notes: tuple[str, ...]
    suites: tuple[SweepSuiteEvaluation, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["project_overrides"] = list(self.project_overrides)
        payload["notes"] = list(self.notes)
        payload["suites"] = [suite.to_dict() for suite in self.suites]
        return payload


@dataclass(frozen=True, slots=True)
class SweepShortlistSummary:
    """Top-level metadata for the shortlist run."""

    generated_at: str
    config_path: str
    base_stage3_config_path: str
    output_root: str
    stage2_checkpoint: str
    run_clean_dev: bool
    corrupted_suite_ids: tuple[str, ...]
    clean_weight: float
    corrupted_weight: float
    eer_weight: float
    min_dcf_weight: float
    configured_candidate_count: int
    executed_candidate_count: int
    max_candidates_budget: int
    budget_notes: tuple[str, ...]
    winner_candidate_id: str
    winner_selection_score: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["corrupted_suite_ids"] = list(self.corrupted_suite_ids)
        payload["budget_notes"] = list(self.budget_notes)
        return payload


@dataclass(frozen=True, slots=True)
class SweepShortlistRunArtifacts:
    """Written shortlist leaderboard and per-candidate results."""

    output_root: str
    report_json_path: str
    report_markdown_path: str
    summary: SweepShortlistSummary
    candidates: tuple[SweepCandidateResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "summary": self.summary.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class _CorruptedSuiteEntry:
    suite_id: str
    family: str
    description: str
    manifest_path: str
    trial_manifest_paths: tuple[str, ...]


def load_campp_sweep_shortlist_report(
    *,
    report_path: Path | str,
) -> SweepShortlistRunArtifacts:
    """Load a previously written shortlist leaderboard JSON report."""

    payload = json.loads(Path(report_path).read_text())
    summary_payload = _require_report_mapping(payload.get("summary"), field_name="summary")
    candidate_payloads = payload.get("candidates")
    if not isinstance(candidate_payloads, list) or not candidate_payloads:
        raise ValueError("Shortlist report must contain a non-empty candidates list.")

    summary = SweepShortlistSummary(
        generated_at=_require_report_string(
            summary_payload.get("generated_at"),
            field_name="summary.generated_at",
        ),
        config_path=_require_report_string(
            summary_payload.get("config_path"),
            field_name="summary.config_path",
        ),
        base_stage3_config_path=_require_report_string(
            summary_payload.get("base_stage3_config_path"),
            field_name="summary.base_stage3_config_path",
        ),
        output_root=_require_report_string(
            summary_payload.get("output_root"),
            field_name="summary.output_root",
        ),
        stage2_checkpoint=_require_report_string(
            summary_payload.get("stage2_checkpoint"),
            field_name="summary.stage2_checkpoint",
        ),
        run_clean_dev=_require_report_bool(
            summary_payload.get("run_clean_dev"),
            field_name="summary.run_clean_dev",
        ),
        corrupted_suite_ids=_require_report_string_tuple(
            summary_payload.get("corrupted_suite_ids"),
            field_name="summary.corrupted_suite_ids",
        ),
        clean_weight=_require_report_float(
            summary_payload.get("clean_weight"),
            field_name="summary.clean_weight",
        ),
        corrupted_weight=_require_report_float(
            summary_payload.get("corrupted_weight"),
            field_name="summary.corrupted_weight",
        ),
        eer_weight=_require_report_float(
            summary_payload.get("eer_weight"),
            field_name="summary.eer_weight",
        ),
        min_dcf_weight=_require_report_float(
            summary_payload.get("min_dcf_weight"),
            field_name="summary.min_dcf_weight",
        ),
        configured_candidate_count=_require_report_int(
            summary_payload.get("configured_candidate_count"),
            field_name="summary.configured_candidate_count",
        ),
        executed_candidate_count=_require_report_int(
            summary_payload.get("executed_candidate_count"),
            field_name="summary.executed_candidate_count",
        ),
        max_candidates_budget=_require_report_int(
            summary_payload.get("max_candidates_budget"),
            field_name="summary.max_candidates_budget",
        ),
        budget_notes=_require_report_string_tuple(
            summary_payload.get("budget_notes"),
            field_name="summary.budget_notes",
        ),
        winner_candidate_id=_require_report_string(
            summary_payload.get("winner_candidate_id"),
            field_name="summary.winner_candidate_id",
        ),
        winner_selection_score=_require_report_float(
            summary_payload.get("winner_selection_score"),
            field_name="summary.winner_selection_score",
        ),
    )
    candidates = tuple(_load_candidate_report_entry(item) for item in candidate_payloads)
    return SweepShortlistRunArtifacts(
        output_root=_require_report_string(payload.get("output_root"), field_name="output_root"),
        report_json_path=_require_report_string(
            payload.get("report_json_path"),
            field_name="report_json_path",
        ),
        report_markdown_path=_require_report_string(
            payload.get("report_markdown_path"),
            field_name="report_markdown_path",
        ),
        summary=summary,
        candidates=candidates,
    )


def run_campp_sweep_shortlist(
    config: CAMPPlusSweepShortlistConfig,
    *,
    config_path: Path | str,
    env_file: Path | str | None = None,
    device_override: str | None = None,
    stage2_checkpoint: Path | str | None = None,
    candidate_ids: tuple[str, ...] | None = None,
    candidate_limit: int | None = None,
) -> SweepShortlistRunArtifacts:
    """Execute the configured shortlist and rank candidates on robust dev."""

    from .stage3_config import load_campp_stage3_config

    base_stage3_config = load_campp_stage3_config(
        config_path=config.base_stage3_config_path,
        env_file=env_file,
        project_overrides=list(config.project_overrides),
    )
    if stage2_checkpoint is not None:
        base_stage3_config = replace(
            base_stage3_config,
            stage3=replace(
                base_stage3_config.stage3,
                stage2_checkpoint=str(stage2_checkpoint),
            ),
        )

    selected_candidates = _select_candidates(
        configured_candidates=config.candidates,
        candidate_ids=candidate_ids,
        candidate_limit=candidate_limit,
    )
    project_root = resolve_project_path(base_stage3_config.project.paths.project_root, ".")
    shortlist_output_root = (
        resolve_project_path(str(project_root), config.output_root) / create_run_id()
    )
    shortlist_output_root.mkdir(parents=True, exist_ok=True)

    corrupted_suites = _load_corrupted_suites(
        project_root=project_root,
        catalog_path=config.corrupted_suites.catalog_path,
        suite_ids=config.corrupted_suites.suite_ids,
    )

    raw_results: list[SweepCandidateResult] = []
    for candidate in selected_candidates:
        candidate_config = _materialize_candidate_config(
            base_config=base_stage3_config,
            candidate=candidate,
            project_root=project_root,
            env_file=env_file,
            shortlist_output_root=shortlist_output_root,
        )
        artifacts = run_campp_stage3(
            candidate_config,
            config_path=config.base_stage3_config_path,
            device_override=device_override,
        )
        suite_results = _evaluate_candidate_suites(
            candidate_id=candidate.candidate_id,
            candidate_config=candidate_config,
            artifacts=artifacts,
            project_root=project_root,
            device_override=device_override,
            corrupted_suites=corrupted_suites,
            run_clean_dev=config.corrupted_suites.run_clean_dev,
        )
        raw_results.append(
            _build_candidate_result(
                candidate=candidate,
                candidate_config=candidate_config,
                artifacts=artifacts,
                suites=suite_results,
                selection=config.selection,
                project_root=project_root,
            )
        )

    if not raw_results:
        raise ValueError("No shortlist candidates were selected for execution.")

    ranked_results = _rank_candidates(raw_results)
    summary = SweepShortlistSummary(
        generated_at=_utc_now(),
        config_path=relative_to_project(Path(config_path), project_root=project_root),
        base_stage3_config_path=relative_to_project(
            resolve_project_path(str(project_root), config.base_stage3_config_path),
            project_root=project_root,
        ),
        output_root=relative_to_project(shortlist_output_root, project_root=project_root),
        stage2_checkpoint=_relative_stage2_checkpoint(
            checkpoint_path=base_stage3_config.stage3.stage2_checkpoint,
            project_root=project_root,
        ),
        run_clean_dev=config.corrupted_suites.run_clean_dev,
        corrupted_suite_ids=tuple(suite.suite_id for suite in corrupted_suites),
        clean_weight=config.selection.clean_weight,
        corrupted_weight=config.selection.corrupted_weight,
        eer_weight=config.selection.eer_weight,
        min_dcf_weight=config.selection.min_dcf_weight,
        configured_candidate_count=len(config.candidates),
        executed_candidate_count=len(ranked_results),
        max_candidates_budget=config.budget.max_candidates,
        budget_notes=config.budget.notes,
        winner_candidate_id=ranked_results[0].candidate_id,
        winner_selection_score=ranked_results[0].selection_score,
    )
    report_json_path = shortlist_output_root / SHORTLIST_REPORT_JSON_NAME
    report_markdown_path = shortlist_output_root / SHORTLIST_REPORT_MARKDOWN_NAME
    artifacts = SweepShortlistRunArtifacts(
        output_root=summary.output_root,
        report_json_path=relative_to_project(report_json_path, project_root=project_root),
        report_markdown_path=relative_to_project(report_markdown_path, project_root=project_root),
        summary=summary,
        candidates=tuple(ranked_results),
    )
    report_json_path.write_text(json.dumps(artifacts.to_dict(), indent=2, sort_keys=True) + "\n")
    report_markdown_path.write_text(
        render_campp_sweep_shortlist_markdown(artifacts) + "\n",
        encoding="utf-8",
    )
    return artifacts


def render_campp_sweep_shortlist_markdown(artifacts: SweepShortlistRunArtifacts) -> str:
    """Render the shortlist leaderboard as markdown."""

    summary = artifacts.summary
    winner = artifacts.candidates[0]
    lines = [
        "# CAM++ Stage-3 Hyperparameter Sweep Shortlist",
        "",
        f"- Output root: `{summary.output_root}`",
        f"- Config: `{summary.config_path}`",
        f"- Base stage-3 config: `{summary.base_stage3_config_path}`",
        f"- Stage-2 checkpoint: `{summary.stage2_checkpoint}`",
        (
            f"- Candidate budget: `{summary.executed_candidate_count}` / "
            f"`{summary.max_candidates_budget}`"
        ),
        (
            "- Ranking objective: "
            f"`{summary.eer_weight:.2f} * weighted_eer + "
            f"{summary.min_dcf_weight:.2f} * weighted_min_dcf`, "
            f"with clean/corrupted weights `{summary.clean_weight:.2f}` / "
            f"`{summary.corrupted_weight:.2f}`"
        ),
        f"- Clean dev included: `{summary.run_clean_dev}`",
        f"- Corrupted suites: `{list(summary.corrupted_suite_ids)}`",
    ]
    for note in summary.budget_notes:
        lines.append(f"- Budget note: {note}")
    lines.extend(
        [
            "",
            "## Winner",
            "",
            f"- Candidate: `{winner.candidate_id}`",
            f"- Selection score: `{winner.selection_score:.6f}`",
            (
                f"- Weighted EER / minDCF: `{winner.weighted_eer:.6f}` / "
                f"`{winner.weighted_min_dcf:.6f}`"
            ),
            f"- Robust EER / minDCF: `{winner.robust_eer:.6f}` / `{winner.robust_min_dcf:.6f}`",
            f"- Clean EER / minDCF: `{winner.clean_eer}` / `{winner.clean_min_dcf}`",
            f"- Run output: `{winner.run_output_root}`",
            f"- Report: `{winner.run_report_path}`",
            "",
            "## Leaderboard",
            "",
            (
                "| Rank | Candidate | Score | Weighted EER | Weighted minDCF | "
                "Robust EER | Clean EER | Pooling | Batch | Crop (s) | Margin |"
            ),
            "|------|-----------|-------|--------------|-----------------|------------|-----------|---------|-------|----------|--------|",
        ]
    )
    for candidate in artifacts.candidates:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(candidate.rank),
                    f"`{candidate.candidate_id}`",
                    f"{candidate.selection_score:.6f}",
                    f"{candidate.weighted_eer:.6f}",
                    f"{candidate.weighted_min_dcf:.6f}",
                    f"{candidate.robust_eer:.6f}",
                    ("-" if candidate.clean_eer is None else f"{candidate.clean_eer:.6f}"),
                    candidate.eval_pooling,
                    str(candidate.effective_batch_size),
                    f"{candidate.crop_start_seconds:.2f}->{candidate.crop_end_seconds:.2f}",
                    f"{candidate.margin_start:.2f}->{candidate.margin_end:.2f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Suite Breakdown", ""])
    for candidate in artifacts.candidates:
        lines.extend(
            [
                f"### `{candidate.candidate_id}`",
                "",
                f"- Description: {candidate.description}",
                f"- Run output: `{candidate.run_output_root}`",
                f"- Run report: `{candidate.run_report_path}`",
                "",
                "| Suite | Family | EER | minDCF | Trials | Report |",
                "|------|--------|-----|--------|--------|--------|",
            ]
        )
        for suite in candidate.suites:
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


def _select_candidates(
    *,
    configured_candidates: tuple[SweepCandidateConfig, ...],
    candidate_ids: tuple[str, ...] | None,
    candidate_limit: int | None,
) -> tuple[SweepCandidateConfig, ...]:
    if candidate_limit is not None and candidate_limit < 1:
        raise ValueError("candidate_limit must be at least 1 when provided.")
    selected = list(configured_candidates)
    if candidate_ids:
        wanted = list(candidate_ids)
        lookup = {candidate.candidate_id: candidate for candidate in configured_candidates}
        missing = [candidate_id for candidate_id in wanted if candidate_id not in lookup]
        if missing:
            raise ValueError(f"Unknown shortlist candidate ids: {missing}")
        selected = [lookup[candidate_id] for candidate_id in wanted]
    if candidate_limit is not None:
        selected = selected[:candidate_limit]
    return tuple(selected)


def _load_candidate_report_entry(payload: object) -> SweepCandidateResult:
    candidate_payload = _require_report_mapping(payload, field_name="candidates[]")
    suite_payloads = candidate_payload.get("suites")
    if not isinstance(suite_payloads, list) or not suite_payloads:
        raise ValueError("candidates[].suites must be a non-empty list.")
    return SweepCandidateResult(
        candidate_id=_require_report_string(
            candidate_payload.get("candidate_id"),
            field_name="candidates[].candidate_id",
        ),
        description=_require_report_string(
            candidate_payload.get("description"),
            field_name="candidates[].description",
        ),
        rank=_require_report_int(candidate_payload.get("rank"), field_name="candidates[].rank"),
        selection_score=_require_report_float(
            candidate_payload.get("selection_score"),
            field_name="candidates[].selection_score",
        ),
        weighted_eer=_require_report_float(
            candidate_payload.get("weighted_eer"),
            field_name="candidates[].weighted_eer",
        ),
        weighted_min_dcf=_require_report_float(
            candidate_payload.get("weighted_min_dcf"),
            field_name="candidates[].weighted_min_dcf",
        ),
        clean_eer=_require_optional_report_float(
            candidate_payload.get("clean_eer"),
            field_name="candidates[].clean_eer",
        ),
        clean_min_dcf=_require_optional_report_float(
            candidate_payload.get("clean_min_dcf"),
            field_name="candidates[].clean_min_dcf",
        ),
        robust_eer=_require_report_float(
            candidate_payload.get("robust_eer"),
            field_name="candidates[].robust_eer",
        ),
        robust_min_dcf=_require_report_float(
            candidate_payload.get("robust_min_dcf"),
            field_name="candidates[].robust_min_dcf",
        ),
        train_batch_size=_require_report_int(
            candidate_payload.get("train_batch_size"),
            field_name="candidates[].train_batch_size",
        ),
        gradient_accumulation_steps=_require_report_int(
            candidate_payload.get("gradient_accumulation_steps"),
            field_name="candidates[].gradient_accumulation_steps",
        ),
        effective_batch_size=_require_report_int(
            candidate_payload.get("effective_batch_size"),
            field_name="candidates[].effective_batch_size",
        ),
        eval_pooling=_require_report_string(
            candidate_payload.get("eval_pooling"),
            field_name="candidates[].eval_pooling",
        ),
        crop_start_seconds=_require_report_float(
            candidate_payload.get("crop_start_seconds"),
            field_name="candidates[].crop_start_seconds",
        ),
        crop_end_seconds=_require_report_float(
            candidate_payload.get("crop_end_seconds"),
            field_name="candidates[].crop_end_seconds",
        ),
        margin_start=_require_report_float(
            candidate_payload.get("margin_start"),
            field_name="candidates[].margin_start",
        ),
        margin_end=_require_report_float(
            candidate_payload.get("margin_end"),
            field_name="candidates[].margin_end",
        ),
        max_epochs=_require_report_int(
            candidate_payload.get("max_epochs"),
            field_name="candidates[].max_epochs",
        ),
        final_train_loss=_require_report_float(
            candidate_payload.get("final_train_loss"),
            field_name="candidates[].final_train_loss",
        ),
        final_train_accuracy=_require_report_float(
            candidate_payload.get("final_train_accuracy"),
            field_name="candidates[].final_train_accuracy",
        ),
        checkpoint_path=_require_report_string(
            candidate_payload.get("checkpoint_path"),
            field_name="candidates[].checkpoint_path",
        ),
        run_output_root=_require_report_string(
            candidate_payload.get("run_output_root"),
            field_name="candidates[].run_output_root",
        ),
        run_report_path=_require_report_string(
            candidate_payload.get("run_report_path"),
            field_name="candidates[].run_report_path",
        ),
        tracking_run_dir=_require_optional_report_string(
            candidate_payload.get("tracking_run_dir"),
            field_name="candidates[].tracking_run_dir",
        ),
        project_overrides=_require_report_string_tuple(
            candidate_payload.get("project_overrides"),
            field_name="candidates[].project_overrides",
        ),
        notes=_require_report_string_tuple(
            candidate_payload.get("notes"),
            field_name="candidates[].notes",
        ),
        suites=tuple(_load_suite_report_entry(item) for item in suite_payloads),
    )


def _load_suite_report_entry(payload: object) -> SweepSuiteEvaluation:
    suite_payload = _require_report_mapping(payload, field_name="candidates[].suites[]")
    return SweepSuiteEvaluation(
        suite_id=_require_report_string(
            suite_payload.get("suite_id"),
            field_name="candidates[].suites[].suite_id",
        ),
        label=_require_report_string(
            suite_payload.get("label"),
            field_name="candidates[].suites[].label",
        ),
        family=_require_report_string(
            suite_payload.get("family"),
            field_name="candidates[].suites[].family",
        ),
        manifest_path=_require_report_string(
            suite_payload.get("manifest_path"),
            field_name="candidates[].suites[].manifest_path",
        ),
        trials_path=_require_report_string(
            suite_payload.get("trials_path"),
            field_name="candidates[].suites[].trials_path",
        ),
        output_root=_require_report_string(
            suite_payload.get("output_root"),
            field_name="candidates[].suites[].output_root",
        ),
        report_markdown_path=_require_report_string(
            suite_payload.get("report_markdown_path"),
            field_name="candidates[].suites[].report_markdown_path",
        ),
        trial_count=_require_report_int(
            suite_payload.get("trial_count"),
            field_name="candidates[].suites[].trial_count",
        ),
        eer=_require_report_float(
            suite_payload.get("eer"),
            field_name="candidates[].suites[].eer",
        ),
        min_dcf=_require_report_float(
            suite_payload.get("min_dcf"),
            field_name="candidates[].suites[].min_dcf",
        ),
        score_gap=_require_optional_report_float(
            suite_payload.get("score_gap"),
            field_name="candidates[].suites[].score_gap",
        ),
    )


def _load_corrupted_suites(
    *,
    project_root: Path,
    catalog_path: str,
    suite_ids: tuple[str, ...],
) -> tuple[_CorruptedSuiteEntry, ...]:
    catalog_file = resolve_project_path(str(project_root), catalog_path)
    payload = json.loads(catalog_file.read_text())
    suites_raw = payload.get("suites")
    if not isinstance(suites_raw, list):
        raise ValueError(
            f"Corrupted suite catalog {catalog_file} does not contain a 'suites' list."
        )
    suites = [
        _CorruptedSuiteEntry(
            suite_id=str(suite.get("suite_id", "")).strip(),
            family=str(suite.get("family", "unknown")).strip(),
            description=str(suite.get("description", "")).strip(),
            manifest_path=str(suite.get("manifest_path", "")).strip(),
            trial_manifest_paths=tuple(
                str(path).strip()
                for path in cast(list[object], suite.get("trial_manifest_paths", []))
            ),
        )
        for suite in suites_raw
        if isinstance(suite, dict)
    ]
    if suite_ids:
        wanted = list(suite_ids)
        lookup = {suite.suite_id: suite for suite in suites}
        missing = [suite_id for suite_id in wanted if suite_id not in lookup]
        if missing:
            raise ValueError(f"Missing corrupted suites in catalog {catalog_file}: {missing}")
        suites = [lookup[suite_id] for suite_id in wanted]
    if not suites:
        raise ValueError(f"No corrupted suites selected from catalog {catalog_file}.")
    return tuple(suites)


def _materialize_candidate_config(
    *,
    base_config: CAMPPlusStage3Config,
    candidate: SweepCandidateConfig,
    project_root: Path,
    env_file: Path | str | None,
    shortlist_output_root: Path,
) -> CAMPPlusStage3Config:
    from kryptonite.config import load_project_config

    merged_project_overrides = (*base_config.project_overrides, *candidate.project_overrides)
    project = load_project_config(
        config_path=base_config.base_config_path,
        overrides=list(merged_project_overrides),
        env_file=env_file,
    )
    data_output_root = (shortlist_output_root / "runs" / candidate.candidate_id).as_posix()
    stage3 = _apply_stage3_overrides(
        stage3=base_config.stage3,
        margin_override=candidate.margin_schedule,
        crop_override=candidate.crop_curriculum,
    )
    return replace(
        base_config,
        project_overrides=tuple(merged_project_overrides),
        project=project,
        data=replace(base_config.data, output_root=data_output_root),
        stage3=stage3,
    )


def _apply_stage3_overrides(
    *,
    stage3: Stage3Config,
    margin_override: SweepMarginScheduleOverride,
    crop_override: SweepCropCurriculumOverride,
) -> Stage3Config:
    margin_schedule = replace(
        stage3.margin_schedule,
        enabled=(
            stage3.margin_schedule.enabled
            if margin_override.enabled is None
            else margin_override.enabled
        ),
        start_margin=(
            stage3.margin_schedule.start_margin
            if margin_override.start_margin is None
            else margin_override.start_margin
        ),
        end_margin=(
            stage3.margin_schedule.end_margin
            if margin_override.end_margin is None
            else margin_override.end_margin
        ),
        ramp_epochs=(
            stage3.margin_schedule.ramp_epochs
            if margin_override.ramp_epochs is None
            else margin_override.ramp_epochs
        ),
    )
    crop_curriculum = replace(
        stage3.crop_curriculum,
        enabled=(
            stage3.crop_curriculum.enabled
            if crop_override.enabled is None
            else crop_override.enabled
        ),
        start_crop_seconds=(
            stage3.crop_curriculum.start_crop_seconds
            if crop_override.start_crop_seconds is None
            else crop_override.start_crop_seconds
        ),
        end_crop_seconds=(
            stage3.crop_curriculum.end_crop_seconds
            if crop_override.end_crop_seconds is None
            else crop_override.end_crop_seconds
        ),
        curriculum_epochs=(
            stage3.crop_curriculum.curriculum_epochs
            if crop_override.curriculum_epochs is None
            else crop_override.curriculum_epochs
        ),
    )
    return replace(stage3, margin_schedule=margin_schedule, crop_curriculum=crop_curriculum)


def _evaluate_candidate_suites(
    *,
    candidate_id: str,
    candidate_config: CAMPPlusStage3Config,
    artifacts: Any,
    project_root: Path,
    device_override: str | None,
    corrupted_suites: tuple[_CorruptedSuiteEntry, ...],
    run_clean_dev: bool,
) -> tuple[SweepSuiteEvaluation, ...]:
    suite_results: list[SweepSuiteEvaluation] = []
    if run_clean_dev:
        suite_results.append(
            _build_clean_suite_evaluation(
                artifacts=artifacts,
                project_root=project_root,
            )
        )

    checkpoint_path = Path(artifacts.checkpoint_path)
    model = CAMPPlusEncoder(candidate_config.model).to(
        resolve_device(device_override or candidate_config.project.runtime.device)
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    evaluation_device = resolve_device(device_override or candidate_config.project.runtime.device)
    audio_request = AudioLoadRequest.from_config(
        candidate_config.project.normalization,
        vad=candidate_config.project.vad,
    )
    feature_request = FbankExtractionRequest.from_config(candidate_config.project.features)

    for suite in corrupted_suites:
        suite_results.append(
            _evaluate_corrupted_suite(
                candidate_id=candidate_id,
                candidate_config=candidate_config,
                suite=suite,
                project_root=project_root,
                artifacts_output_root=Path(artifacts.output_root),
                model=model,
                device=evaluation_device,
                audio_request=audio_request,
                feature_request=feature_request,
            )
        )
    return tuple(suite_results)


def _build_clean_suite_evaluation(*, artifacts: Any, project_root: Path) -> SweepSuiteEvaluation:
    verification_report = artifacts.verification_report
    if verification_report is None:
        raise ValueError("Stage-3 run did not produce the clean-dev verification report.")
    metrics = verification_report.summary.metrics
    return SweepSuiteEvaluation(
        suite_id="clean_dev",
        label="clean_dev",
        family="clean",
        manifest_path=str(artifacts.training_summary.dev_manifest),
        trials_path=relative_to_project(Path(artifacts.trials_path), project_root=project_root),
        output_root=relative_to_project(Path(artifacts.output_root), project_root=project_root),
        report_markdown_path=relative_to_project(
            Path(verification_report.report_markdown_path),
            project_root=project_root,
        ),
        trial_count=metrics.trial_count,
        eer=metrics.eer,
        min_dcf=metrics.min_dcf,
        score_gap=artifacts.score_summary.score_gap,
    )


def _evaluate_corrupted_suite(
    *,
    candidate_id: str,
    candidate_config: CAMPPlusStage3Config,
    suite: _CorruptedSuiteEntry,
    project_root: Path,
    artifacts_output_root: Path,
    model: CAMPPlusEncoder,
    device: torch.device,
    audio_request: AudioLoadRequest,
    feature_request: FbankExtractionRequest,
) -> SweepSuiteEvaluation:
    suite_output_root = artifacts_output_root / "robust_dev" / suite.suite_id
    suite_output_root.mkdir(parents=True, exist_ok=True)
    suite_rows = load_manifest_rows(
        suite.manifest_path,
        project_root=project_root,
        limit=candidate_config.data.max_dev_rows,
    )
    embedding_summary, metadata_rows = export_dev_embeddings(
        output_root=suite_output_root,
        model=model,
        rows=suite_rows,
        manifest_path=suite.manifest_path,
        project_root=project_root,
        audio_request=audio_request,
        feature_request=feature_request,
        chunking=candidate_config.project.chunking,
        device=device,
        embedding_source=f"campp_stage3_sweep:{candidate_id}:{suite.suite_id}",
    )
    trials_path, trial_rows = _resolve_suite_trials(
        suite=suite,
        output_root=suite_output_root,
        metadata_rows=metadata_rows,
        project_root=project_root,
    )
    build_default_cohort_bank(
        output_root=suite_output_root,
        embedding_summary=embedding_summary,
        train_manifest_path=candidate_config.data.train_manifest,
        trials_path=trials_path,
        project_root=project_root,
    )
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
    return SweepSuiteEvaluation(
        suite_id=suite.suite_id,
        label=suite.suite_id,
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


def _resolve_suite_trials(
    *,
    suite: _CorruptedSuiteEntry,
    output_root: Path,
    metadata_rows: list[dict[str, Any]],
    project_root: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    if not suite.trial_manifest_paths:
        return load_or_generate_trials(
            output_root=output_root,
            configured_trials_manifest=None,
            metadata_rows=metadata_rows,
            project_root=project_root,
        )
    merged_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for trial_manifest_path in suite.trial_manifest_paths:
        trial_path = resolve_project_path(str(project_root), trial_manifest_path)
        for row in load_verification_trial_rows(trial_path):
            key = _trial_identity(row)
            if key in seen:
                continue
            seen.add(key)
            merged_rows.append(dict(row))
    if not merged_rows:
        raise ValueError(f"Suite {suite.suite_id!r} does not contain any trial rows.")
    trials_path = output_root / SUITE_TRIALS_FILE_NAME
    trials_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in merged_rows),
        encoding="utf-8",
    )
    return trials_path, merged_rows


def _trial_identity(row: dict[str, Any]) -> tuple[str, str, int]:
    left_id = str(row.get("left_id", row.get("left_audio", "")))
    right_id = str(row.get("right_id", row.get("right_audio", "")))
    label = int(row.get("label", 0))
    return left_id, right_id, label


def _build_candidate_result(
    *,
    candidate: SweepCandidateConfig,
    candidate_config: CAMPPlusStage3Config,
    artifacts: Any,
    suites: tuple[SweepSuiteEvaluation, ...],
    selection: SweepSelectionConfig,
    project_root: Path,
) -> SweepCandidateResult:
    clean_suite = next((suite for suite in suites if suite.suite_id == "clean_dev"), None)
    robust_suites = tuple(suite for suite in suites if suite.suite_id != "clean_dev")
    if not robust_suites:
        raise ValueError("Shortlist ranking requires at least one corrupted dev suite result.")

    robust_eer = _mean_metric(tuple(suite.eer for suite in robust_suites))
    robust_min_dcf = _mean_metric(tuple(suite.min_dcf for suite in robust_suites))
    clean_weight, corrupted_weight = _selection_weights(
        selection, has_clean=clean_suite is not None
    )
    weighted_eer = (
        clean_weight * (robust_eer if clean_suite is None else clean_suite.eer)
        + corrupted_weight * robust_eer
    )
    weighted_min_dcf = (
        clean_weight * (robust_min_dcf if clean_suite is None else clean_suite.min_dcf)
        + corrupted_weight * robust_min_dcf
    )
    selection_score = (
        selection.eer_weight * weighted_eer + selection.min_dcf_weight * weighted_min_dcf
    )

    final_epoch = artifacts.training_summary.epochs[-1]
    return SweepCandidateResult(
        candidate_id=candidate.candidate_id,
        description=candidate.description,
        rank=0,
        selection_score=round(selection_score, 6),
        weighted_eer=round(weighted_eer, 6),
        weighted_min_dcf=round(weighted_min_dcf, 6),
        clean_eer=(None if clean_suite is None else clean_suite.eer),
        clean_min_dcf=(None if clean_suite is None else clean_suite.min_dcf),
        robust_eer=round(robust_eer, 6),
        robust_min_dcf=round(robust_min_dcf, 6),
        train_batch_size=candidate_config.project.training.batch_size,
        gradient_accumulation_steps=candidate_config.optimization.gradient_accumulation_steps,
        effective_batch_size=(
            candidate_config.project.training.batch_size
            * candidate_config.optimization.gradient_accumulation_steps
        ),
        eval_pooling=candidate_config.project.chunking.eval_pooling,
        crop_start_seconds=candidate_config.stage3.crop_curriculum.start_crop_seconds,
        crop_end_seconds=candidate_config.stage3.crop_curriculum.end_crop_seconds,
        margin_start=(
            candidate_config.stage3.margin_schedule.start_margin
            if candidate_config.stage3.margin_schedule.enabled
            else candidate_config.objective.margin
        ),
        margin_end=(
            candidate_config.stage3.margin_schedule.end_margin
            if candidate_config.stage3.margin_schedule.enabled
            else candidate_config.objective.margin
        ),
        max_epochs=candidate_config.project.training.max_epochs,
        final_train_loss=final_epoch.mean_loss,
        final_train_accuracy=final_epoch.accuracy,
        checkpoint_path=relative_to_project(
            Path(artifacts.checkpoint_path), project_root=project_root
        ),
        run_output_root=relative_to_project(Path(artifacts.output_root), project_root=project_root),
        run_report_path=relative_to_project(Path(artifacts.report_path), project_root=project_root),
        tracking_run_dir=(
            None
            if artifacts.tracking_run_dir is None
            else relative_to_project(Path(artifacts.tracking_run_dir), project_root=project_root)
        ),
        project_overrides=tuple(candidate_config.project_overrides),
        notes=candidate.notes,
        suites=suites,
    )


def _selection_weights(selection: SweepSelectionConfig, *, has_clean: bool) -> tuple[float, float]:
    if has_clean:
        total = selection.clean_weight + selection.corrupted_weight
        return selection.clean_weight / total, selection.corrupted_weight / total
    return 0.0, 1.0


def _rank_candidates(candidates: list[SweepCandidateResult]) -> list[SweepCandidateResult]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.selection_score,
            candidate.weighted_eer,
            candidate.weighted_min_dcf,
            candidate.candidate_id,
        ),
    )
    return [replace(candidate, rank=index + 1) for index, candidate in enumerate(ordered)]


def _mean_metric(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("Cannot average an empty metric collection.")
    return sum(values) / len(values)


def _relative_stage2_checkpoint(*, checkpoint_path: str, project_root: Path) -> str:
    return relative_to_project(
        resolve_project_path(str(project_root), checkpoint_path),
        project_root=project_root,
    )


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _require_report_mapping(payload: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must be an object.")
    return cast(dict[str, object], payload)


def _require_report_string(payload: object, *, field_name: str) -> str:
    if not isinstance(payload, str) or not payload.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return payload


def _require_optional_report_string(payload: object, *, field_name: str) -> str | None:
    if payload is None:
        return None
    return _require_report_string(payload, field_name=field_name)


def _require_report_bool(payload: object, *, field_name: str) -> bool:
    if not isinstance(payload, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return payload


def _require_report_int(payload: object, *, field_name: str) -> int:
    if not isinstance(payload, int):
        raise ValueError(f"{field_name} must be an integer.")
    return payload


def _require_report_float(payload: object, *, field_name: str) -> float:
    if isinstance(payload, bool) or not isinstance(payload, int | float):
        raise ValueError(f"{field_name} must be a number.")
    return float(payload)


def _require_optional_report_float(payload: object, *, field_name: str) -> float | None:
    if payload is None:
        return None
    return _require_report_float(payload, field_name=field_name)


def _require_report_string_tuple(payload: object, *, field_name: str) -> tuple[str, ...]:
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise ValueError(f"{field_name} must be a list of strings.")
    values: list[str] = []
    for index, item in enumerate(payload):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{index}] must be a string.")
        values.append(item)
    return tuple(values)


__all__ = [
    "SHORTLIST_REPORT_JSON_NAME",
    "SHORTLIST_REPORT_MARKDOWN_NAME",
    "SweepCandidateResult",
    "SweepShortlistRunArtifacts",
    "SweepShortlistSummary",
    "SweepSuiteEvaluation",
    "load_campp_sweep_shortlist_report",
    "render_campp_sweep_shortlist_markdown",
    "run_campp_sweep_shortlist",
]
