"""Dataset manifests, validation, loading, and preprocessing."""

from .audio_loader import (
    AudioLoadRequest,
    LoadedAudio,
    LoadedManifestAudio,
    iter_manifest_audio,
    load_audio,
    load_manifest_audio,
)
from .schema import (
    MANIFEST_RECORD_TYPE,
    MANIFEST_SCHEMA_VERSION,
    ManifestRow,
    ManifestValidationError,
    ManifestValidationIssue,
    normalize_manifest_entry,
    validate_manifest_entry,
)
from .validation import (
    ManifestRowIssue,
    ManifestValidationReport,
    SkippedManifest,
    build_manifest_validation_report,
)

__all__ = [
    "AudioLoadRequest",
    "LoadedAudio",
    "LoadedManifestAudio",
    "MANIFEST_RECORD_TYPE",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestRow",
    "ManifestRowIssue",
    "ManifestValidationError",
    "ManifestValidationIssue",
    "ManifestValidationReport",
    "SkippedManifest",
    "build_manifest_validation_report",
    "iter_manifest_audio",
    "load_audio",
    "load_manifest_audio",
    "normalize_manifest_entry",
    "validate_manifest_entry",
]
