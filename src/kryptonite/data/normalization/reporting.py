"""Rendering helpers for audio normalization reports."""

from .models import AudioNormalizationSummary


def render_summary_markdown(summary: AudioNormalizationSummary) -> str:
    issue_lines = [
        f"- `{issue_code}`: {count}"
        for issue_code, count in summary.quarantine_issue_counts.items()
    ] or ["- none"]
    return "\n".join(
        [
            "# Audio Normalization Report",
            "",
            "## Scope",
            "",
            f"- dataset: `{summary.dataset}`",
            f"- source manifests root: `{summary.source_manifests_root}`",
            f"- output root: `{summary.output_root}`",
            "",
            "## Policy",
            "",
            f"- normalization profile: `{summary.policy.normalization_profile}`",
            f"- peak headroom: `{summary.policy.peak_headroom_db:.2f} dB`",
            f"- dc offset threshold: `{summary.policy.dc_offset_threshold:.4f}`",
            f"- clipped sample threshold: `{summary.policy.clipped_sample_threshold:.4f}`",
            f"- loudness mode: `{summary.policy.loudness_mode}`",
            f"- target loudness: `{summary.policy.target_loudness_dbfs:.2f} dBFS`",
            f"- max loudness gain: `{summary.policy.max_loudness_gain_db:.2f} dB`",
            f"- max loudness attenuation: `{summary.policy.max_loudness_attenuation_db:.2f} dB`",
            "",
            "## Results",
            "",
            f"- source manifests: `{summary.source_manifest_count}`",
            f"- source rows: `{summary.source_row_count}`",
            f"- normalized rows: `{summary.normalized_row_count}`",
            f"- normalized audio files: `{summary.normalized_audio_count}`",
            f"- generated quarantine rows: `{summary.generated_quarantine_row_count}`",
            f"- carried quarantine rows: `{summary.carried_quarantine_row_count}`",
            f"- resampled rows: `{summary.resampled_row_count}`",
            f"- downmixed rows: `{summary.downmixed_row_count}`",
            f"- dc-offset fixes: `{summary.dc_offset_fixed_row_count}`",
            f"- peak-scaled rows: `{summary.peak_scaled_row_count}`",
            f"- loudness-applied rows: `{summary.loudness_applied_row_count}`",
            f"- loudness gain-capped rows: `{summary.loudness_gain_limited_row_count}`",
            f"- loudness peak-limited rows: `{summary.loudness_peak_limited_row_count}`",
            (
                "- loudness degradation check failures: "
                f"`{summary.loudness_degradation_failed_row_count}`"
            ),
            f"- source clipping detections: `{summary.source_clipping_row_count}`",
            f"- mean source loudness: `{_format_optional_dbfs(summary.mean_source_rms_dbfs)}`",
            (
                "- mean normalized loudness: "
                f"`{_format_optional_dbfs(summary.mean_normalized_rms_dbfs)}`"
            ),
            "",
            "## Quarantine Issues",
            "",
            *issue_lines,
            "",
        ]
    )


def _format_optional_dbfs(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} dBFS"
