"""Render and write the final benchmark pack artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from .final_benchmark_pack_models import (
    FINAL_BENCHMARK_PACK_CANDIDATES_JSONL_NAME,
    FINAL_BENCHMARK_PACK_JSON_NAME,
    FINAL_BENCHMARK_PACK_MARKDOWN_NAME,
    FINAL_BENCHMARK_PACK_PAIRWISE_JSONL_NAME,
    FinalBenchmarkPackReport,
    WrittenFinalBenchmarkPack,
)


def write_final_benchmark_pack(report: FinalBenchmarkPackReport) -> WrittenFinalBenchmarkPack:
    """Write JSON/Markdown/JSONL artifacts for the staged benchmark pack."""

    output_root = Path(report.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / FINAL_BENCHMARK_PACK_JSON_NAME
    markdown_path = output_root / FINAL_BENCHMARK_PACK_MARKDOWN_NAME
    candidate_jsonl_path = output_root / FINAL_BENCHMARK_PACK_CANDIDATES_JSONL_NAME
    pairwise_jsonl_path = output_root / FINAL_BENCHMARK_PACK_PAIRWISE_JSONL_NAME

    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_final_benchmark_pack_markdown(report) + "\n",
        encoding="utf-8",
    )
    candidate_jsonl_path.write_text(
        "".join(
            json.dumps(candidate.to_dict(), sort_keys=True) + "\n"
            for candidate in report.candidates
        ),
        encoding="utf-8",
    )
    pairwise_jsonl_path.write_text(
        "".join(
            json.dumps(comparison.to_dict(), sort_keys=True) + "\n"
            for comparison in report.pairwise_comparisons
        ),
        encoding="utf-8",
    )
    return WrittenFinalBenchmarkPack(
        output_root=str(output_root),
        report_json_path=str(json_path),
        report_markdown_path=str(markdown_path),
        candidate_jsonl_path=str(candidate_jsonl_path),
        pairwise_jsonl_path=str(pairwise_jsonl_path),
        summary=report.summary,
    )


def render_final_benchmark_pack_markdown(report: FinalBenchmarkPackReport) -> str:
    """Render one human-readable release benchmark pack summary."""

    lines = [f"# {report.title}", ""]
    if report.summary_text:
        lines.extend([report.summary_text, ""])
    lines.extend(
        [
            "## Summary",
            "",
            f"- Candidates: `{report.summary.candidate_count}`",
            f"- Pairwise comparisons: `{report.summary.pairwise_comparison_count}`",
            f"- Best EER candidate: `{report.summary.best_eer_candidate_id or '-'}`",
            f"- Best minDCF candidate: `{report.summary.best_min_dcf_candidate_id or '-'}`",
            f"- Lowest latency candidate: `{report.summary.lowest_latency_candidate_id or '-'}`",
            (
                "- Lowest process RSS candidate: "
                f"`{report.summary.lowest_process_rss_candidate_id or '-'}`"
            ),
            (
                "- Lowest CUDA allocated candidate: "
                f"`{report.summary.lowest_cuda_allocated_candidate_id or '-'}`"
            ),
            "",
            "## Candidate Overview",
            "",
            _markdown_table(
                headers=[
                    "Candidate",
                    "Family",
                    "EER",
                    "MinDCF",
                    "Balanced thr",
                    "Largest burst",
                    "ms/audio",
                    "Peak RSS MiB",
                    "Peak CUDA MiB",
                    "Model version",
                ],
                rows=[
                    [
                        candidate.label,
                        candidate.family,
                        _format_float(candidate.quality.eer, digits=6),
                        _format_float(candidate.quality.min_dcf, digits=6),
                        _format_float(candidate.quality.balanced_threshold, digits=6),
                        str(candidate.operational.largest_validated_batch_size),
                        _format_float(
                            candidate.operational.mean_ms_per_audio_at_largest_batch,
                            digits=6,
                        ),
                        _format_float(candidate.operational.peak_process_rss_mib, digits=3),
                        _format_float(candidate.operational.peak_cuda_allocated_mib, digits=3),
                        candidate.bundle.model_version,
                    ]
                    for candidate in report.candidates
                ],
            ),
        ]
    )

    if report.pairwise_comparisons:
        lines.extend(
            [
                "",
                "## Pairwise Comparisons",
                "",
                _markdown_table(
                    headers=[
                        "Left",
                        "Right",
                        "dEER",
                        "dMinDCF",
                        "dLatency ms/audio",
                        "dRSS MiB",
                        "dCUDA MiB",
                        "Better quality",
                    ],
                    rows=[
                        [
                            comparison.left_candidate_id,
                            comparison.right_candidate_id,
                            _format_float(comparison.eer_delta_right_minus_left, digits=6),
                            _format_float(comparison.min_dcf_delta_right_minus_left, digits=6),
                            _format_float(
                                comparison.latency_delta_ms_per_audio_right_minus_left,
                                digits=6,
                            ),
                            _format_float(
                                comparison.process_rss_delta_mib_right_minus_left,
                                digits=3,
                            ),
                            _format_float(
                                comparison.cuda_allocated_delta_mib_right_minus_left,
                                digits=3,
                            ),
                            comparison.better_quality_candidate_id or "-",
                        ]
                        for comparison in report.pairwise_comparisons
                    ],
                ),
            ]
        )

    lines.extend(["", "## Source Artifacts", ""])
    if report.source_config_artifact is not None:
        lines.append(
            "- Pack config: "
            f"`{report.source_config_artifact.copied_path}` "
            f"(sha256 `{report.source_config_artifact.sha256}`)"
        )
    for candidate in report.candidates:
        lines.extend(["", f"### {candidate.label}", ""])
        lines.extend(
            f"- {artifact.kind}: `{artifact.copied_path}` (sha256 `{artifact.sha256}`)"
            for artifact in candidate.source_artifacts
        )
        lines.extend(f"- Note: {note}" for note in candidate.notes)
    lines.extend(f"- Note: {note}" for note in report.notes)
    return "\n".join(lines)


def _markdown_table(*, headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _format_float(value: float | None, *, digits: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


__all__ = [
    "render_final_benchmark_pack_markdown",
    "write_final_benchmark_pack",
]
