"""Coverage report generation for augmentation scheduling."""

from __future__ import annotations

import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from kryptonite.config import AugmentationSchedulerConfig, SilenceAugmentationConfig

from .io import load_augmentation_catalog, resolve_bank_manifest_paths
from .models import (
    AUGMENTATION_FAMILY_ORDER,
    AUGMENTATION_INTENSITY_ORDER,
    NON_CLEAN_INTENSITY_ORDER,
    AugmentationSchedulerReport,
    AugmentationSchedulerSummary,
    EpochCoverageSummary,
    WrittenAugmentationSchedulerArtifacts,
)
from .scheduler import AugmentationScheduler

REPORT_JSON_NAME = "augmentation_scheduler_report.json"
REPORT_MARKDOWN_NAME = "augmentation_scheduler_report.md"
EPOCH_ROWS_NAME = "augmentation_scheduler_epochs.jsonl"


def build_augmentation_scheduler_report(
    *,
    project_root: Path | str,
    scheduler_config: AugmentationSchedulerConfig,
    silence_config: SilenceAugmentationConfig,
    total_epochs: int,
    samples_per_epoch: int,
    seed: int,
    noise_manifest_path: Path | str | None = None,
    room_config_manifest_path: Path | str | None = None,
    distance_manifest_path: Path | str | None = None,
    codec_manifest_path: Path | str | None = None,
) -> AugmentationSchedulerReport:
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if samples_per_epoch <= 0:
        raise ValueError("samples_per_epoch must be positive")

    project_root_path = Path(project_root)
    manifest_paths = resolve_bank_manifest_paths(
        project_root=project_root_path,
        noise_manifest_path=noise_manifest_path,
        room_config_manifest_path=room_config_manifest_path,
        distance_manifest_path=distance_manifest_path,
        codec_manifest_path=codec_manifest_path,
    )
    catalog = load_augmentation_catalog(
        manifest_paths=manifest_paths,
        silence_config=silence_config,
    )
    if scheduler_config.enabled and not catalog.available_families:
        raise ValueError("No augmentation families are available for the scheduler.")

    scheduler = AugmentationScheduler(
        config=scheduler_config,
        catalog=catalog,
        total_epochs=total_epochs,
    )
    rng = random.Random(seed)
    epoch_summaries: list[EpochCoverageSummary] = []
    stage_epoch_counts: Counter[str] = Counter()
    overall_intensity_counts: Counter[str] = Counter()
    overall_family_counts: Counter[str] = Counter()
    overall_severity_counts: Counter[str] = Counter()

    for epoch_index in range(1, total_epochs + 1):
        plan = scheduler.plan_for_epoch(epoch_index)
        stage_epoch_counts[plan.stage] += 1
        intensity_counts: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        severity_counts: Counter[str] = Counter()
        candidate_labels: Counter[str] = Counter()

        for _ in range(samples_per_epoch):
            recipe = scheduler.sample_recipe(epoch_index=epoch_index, rng=rng)
            intensity_counts[recipe.intensity] += 1
            overall_intensity_counts[recipe.intensity] += 1
            for augmentation in recipe.augmentations:
                family_counts[augmentation.family] += 1
                severity_counts[augmentation.severity] += 1
                overall_family_counts[augmentation.family] += 1
                overall_severity_counts[augmentation.severity] += 1
                candidate_labels[augmentation.label] += 1

        epoch_summaries.append(
            EpochCoverageSummary(
                epoch_index=epoch_index,
                stage=plan.stage,
                sample_count=samples_per_epoch,
                clean_sample_count=intensity_counts.get("clean", 0),
                max_augmentations_per_sample=plan.max_augmentations_per_sample,
                intensity_probabilities=dict(plan.intensity_probabilities),
                empirical_intensity_ratios={
                    intensity: round(intensity_counts.get(intensity, 0) / samples_per_epoch, 6)
                    for intensity in AUGMENTATION_INTENSITY_ORDER
                },
                family_counts={
                    family: family_counts.get(family, 0) for family in AUGMENTATION_FAMILY_ORDER
                },
                severity_counts={
                    severity: severity_counts.get(severity, 0)
                    for severity in NON_CLEAN_INTENSITY_ORDER
                },
                top_augmentations=tuple(
                    f"{label} ({count})" for label, count in candidate_labels.most_common(5)
                ),
            )
        )

    summary = AugmentationSchedulerSummary(
        total_epochs=total_epochs,
        samples_per_epoch=samples_per_epoch,
        candidate_counts_by_family=dict(catalog.candidate_counts_by_family),
        missing_families=catalog.missing_families,
        stage_epoch_counts=dict(stage_epoch_counts),
        overall_intensity_counts=dict(overall_intensity_counts),
        overall_family_counts=dict(overall_family_counts),
        overall_severity_counts=dict(overall_severity_counts),
    )
    return AugmentationSchedulerReport(
        generated_at=datetime.now(tz=UTC).isoformat(),
        project_root=str(project_root_path),
        seed=seed,
        manifest_paths=manifest_paths,
        scheduler_config=scheduler_config,
        silence_config=silence_config,
        catalog=catalog,
        epochs=tuple(epoch_summaries),
        summary=summary,
    )


def write_augmentation_scheduler_report(
    *,
    report: AugmentationSchedulerReport,
    output_root: Path | str,
) -> WrittenAugmentationSchedulerArtifacts:
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    json_path = output_root_path / REPORT_JSON_NAME
    markdown_path = output_root_path / REPORT_MARKDOWN_NAME
    epochs_path = output_root_path / EPOCH_ROWS_NAME

    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_augmentation_scheduler_markdown(report))
    epochs_path.write_text(
        "".join(json.dumps(epoch.to_dict(), sort_keys=True) + "\n" for epoch in report.epochs)
    )

    return WrittenAugmentationSchedulerArtifacts(
        output_root=str(output_root_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
        epochs_path=str(epochs_path),
    )


def render_augmentation_scheduler_markdown(report: AugmentationSchedulerReport) -> str:
    lines = [
        "# Augmentation Scheduler Report",
        "",
        f"- generated at: `{report.generated_at}`",
        f"- project root: `{report.project_root}`",
        f"- seed: `{report.seed}`",
        f"- total epochs: `{report.summary.total_epochs}`",
        f"- samples per epoch: `{report.summary.samples_per_epoch}`",
        "",
        "## Scheduler Config",
        "",
        f"- enabled: `{report.scheduler_config.enabled}`",
        f"- warmup epochs: `{report.scheduler_config.warmup_epochs}`",
        f"- ramp epochs: `{report.scheduler_config.ramp_epochs}`",
        (
            "- max augmentations per sample: "
            f"`{report.scheduler_config.max_augmentations_per_sample}`"
        ),
        (
            "- probability start: "
            f"`clean={report.scheduler_config.clean_probability_start:.2f}`, "
            f"`light={report.scheduler_config.light_probability_start:.2f}`, "
            f"`medium={report.scheduler_config.medium_probability_start:.2f}`, "
            f"`heavy={report.scheduler_config.heavy_probability_start:.2f}`"
        ),
        (
            "- probability end: "
            f"`clean={report.scheduler_config.clean_probability_end:.2f}`, "
            f"`light={report.scheduler_config.light_probability_end:.2f}`, "
            f"`medium={report.scheduler_config.medium_probability_end:.2f}`, "
            f"`heavy={report.scheduler_config.heavy_probability_end:.2f}`"
        ),
        "",
        "## Manifest Inputs",
        "",
        f"- noise manifest: `{report.manifest_paths.noise_manifest_path}`",
        f"- room-config manifest: `{report.manifest_paths.room_config_manifest_path}`",
        f"- distance manifest: `{report.manifest_paths.distance_manifest_path}`",
        f"- codec manifest: `{report.manifest_paths.codec_manifest_path}`",
        "",
        "## Available Coverage",
        "",
    ]
    for family in AUGMENTATION_FAMILY_ORDER:
        lines.append(
            f"- `{family}` candidates: `{report.summary.candidate_counts_by_family.get(family, 0)}`"
        )
    lines.extend(
        [
            f"- missing families: `{', '.join(report.summary.missing_families) or 'none'}`",
            "",
            "## Overall Sample Mix",
            "",
            (
                "- intensity counts: "
                + ", ".join(
                    f"`{intensity}={report.summary.overall_intensity_counts.get(intensity, 0)}`"
                    for intensity in AUGMENTATION_INTENSITY_ORDER
                )
            ),
            (
                "- family counts: "
                + ", ".join(
                    f"`{family}={report.summary.overall_family_counts.get(family, 0)}`"
                    for family in AUGMENTATION_FAMILY_ORDER
                )
            ),
            (
                "- severity counts: "
                + ", ".join(
                    f"`{severity}={report.summary.overall_severity_counts.get(severity, 0)}`"
                    for severity in NON_CLEAN_INTENSITY_ORDER
                )
            ),
            "",
            "## Epoch Coverage",
            "",
        ]
    )
    for epoch in report.epochs:
        lines.extend(
            [
                f"### Epoch {epoch.epoch_index}",
                "",
                f"- stage: `{epoch.stage}`",
                f"- max augmentations per sample: `{epoch.max_augmentations_per_sample}`",
                (
                    "- target intensity ratios: "
                    + ", ".join(
                        f"`{intensity}={epoch.intensity_probabilities[intensity]:.2f}`"
                        for intensity in AUGMENTATION_INTENSITY_ORDER
                    )
                ),
                (
                    "- empirical intensity ratios: "
                    + ", ".join(
                        f"`{intensity}={epoch.empirical_intensity_ratios[intensity]:.2f}`"
                        for intensity in AUGMENTATION_INTENSITY_ORDER
                    )
                ),
                (
                    "- family coverage: "
                    + (", ".join(f"`{family}`" for family in epoch.family_coverage) or "`none`")
                ),
                (
                    "- severity coverage: "
                    + (
                        ", ".join(f"`{severity}`" for severity in epoch.severity_coverage)
                        or "`none`"
                    )
                ),
                (
                    "- top augmentations: "
                    + (", ".join(f"`{item}`" for item in epoch.top_augmentations) or "`none`")
                ),
                "",
            ]
        )
    return "\n".join(lines)
