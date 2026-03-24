"""Curriculum augmentation scheduler and coverage reporting."""

from .io import load_augmentation_catalog, resolve_bank_manifest_paths
from .models import (
    AUGMENTATION_FAMILY_ORDER,
    AUGMENTATION_INTENSITY_ORDER,
    NON_CLEAN_INTENSITY_ORDER,
    AugmentationCandidate,
    AugmentationCatalog,
    AugmentationFamily,
    AugmentationIntensity,
    AugmentationSchedulerReport,
    AugmentationSchedulerSummary,
    BankManifestPaths,
    EpochAugmentationPlan,
    EpochCoverageSummary,
    ScheduledAugmentation,
    ScheduledSampleRecipe,
    WrittenAugmentationSchedulerArtifacts,
)
from .reporting import (
    build_augmentation_scheduler_report,
    render_augmentation_scheduler_markdown,
    write_augmentation_scheduler_report,
)
from .scheduler import AugmentationScheduler

__all__ = [
    "AUGMENTATION_FAMILY_ORDER",
    "AUGMENTATION_INTENSITY_ORDER",
    "NON_CLEAN_INTENSITY_ORDER",
    "AugmentationFamily",
    "AugmentationIntensity",
    "AugmentationCandidate",
    "AugmentationCatalog",
    "AugmentationScheduler",
    "AugmentationSchedulerReport",
    "AugmentationSchedulerSummary",
    "BankManifestPaths",
    "EpochAugmentationPlan",
    "EpochCoverageSummary",
    "ScheduledAugmentation",
    "ScheduledSampleRecipe",
    "WrittenAugmentationSchedulerArtifacts",
    "build_augmentation_scheduler_report",
    "load_augmentation_catalog",
    "render_augmentation_scheduler_markdown",
    "resolve_bank_manifest_paths",
    "write_augmentation_scheduler_report",
]
