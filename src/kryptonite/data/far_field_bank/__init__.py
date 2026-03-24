"""Reproducible far-field and distance simulation presets and reporting."""

from .models import (
    ALLOWED_DISTANCE_FIELDS,
    MANIFEST_JSONL_NAME,
    PROBE_AUDIO_NAME,
    REPORT_JSON_NAME,
    REPORT_MARKDOWN_NAME,
    DistanceField,
    FarFieldAudioMetrics,
    FarFieldBankEntry,
    FarFieldBankPlan,
    FarFieldBankReport,
    FarFieldBankSummary,
    FarFieldKernelMetrics,
    FarFieldProbeSettings,
    FarFieldRenderSettings,
    FarFieldSimulationPreset,
    WrittenFarFieldArtifacts,
)
from .plan import load_far_field_bank_plan
from .reporting import (
    build_far_field_bank,
    render_far_field_bank_markdown,
    write_far_field_bank_report,
)

__all__ = [
    "ALLOWED_DISTANCE_FIELDS",
    "DistanceField",
    "FarFieldAudioMetrics",
    "FarFieldBankEntry",
    "FarFieldBankPlan",
    "FarFieldBankReport",
    "FarFieldBankSummary",
    "FarFieldKernelMetrics",
    "FarFieldProbeSettings",
    "FarFieldRenderSettings",
    "FarFieldSimulationPreset",
    "MANIFEST_JSONL_NAME",
    "PROBE_AUDIO_NAME",
    "REPORT_JSON_NAME",
    "REPORT_MARKDOWN_NAME",
    "WrittenFarFieldArtifacts",
    "build_far_field_bank",
    "load_far_field_bank_plan",
    "render_far_field_bank_markdown",
    "write_far_field_bank_report",
]
