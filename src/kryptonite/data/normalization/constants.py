"""Constants for normalized manifest bundle layout."""

DATA_MANIFEST_PRIORITY: tuple[str, ...] = (
    "all_manifest.jsonl",
    "train_manifest.jsonl",
    "dev_manifest.jsonl",
)
TRIAL_FIELD_ORDER: tuple[str, ...] = ("label", "left_audio", "right_audio")
QUARANTINE_MANIFEST_NAME = "quarantine_manifest.jsonl"
REPORT_JSON_NAME = "audio_normalization_report.json"
REPORT_MARKDOWN_NAME = "audio_normalization_report.md"
