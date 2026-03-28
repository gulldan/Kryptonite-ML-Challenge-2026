"""Rendering and file-writing helpers for backend benchmark reports."""

from __future__ import annotations

import json
from pathlib import Path

from .backend_benchmark_models import (
    BACKEND_BENCHMARK_REPORT_JSON_NAME,
    BACKEND_BENCHMARK_REPORT_MARKDOWN_NAME,
    BACKEND_BENCHMARK_WORKLOAD_ROWS_NAME,
    BackendBenchmarkPlotAsset,
    BackendBenchmarkReport,
    BackendBenchmarkWorkloadResult,
    WrittenBackendBenchmarkReport,
)

_BACKEND_COLORS: dict[str, str] = {
    "torch": "#2563eb",
    "onnxruntime": "#059669",
    "tensorrt": "#d97706",
}


def render_backend_benchmark_markdown(report: BackendBenchmarkReport) -> str:
    lines = [
        f"# {report.title}",
        "",
        report.summary_text.strip(),
        "",
        "## Summary",
        "",
        f"- Status: `{'pass' if report.summary.passed else 'fail'}`",
        f"- Report id: `{report.report_id}`",
        f"- Generated at: `{report.generated_at_utc}`",
        f"- Model version: `{report.model_version or 'unknown'}`",
        (
            "- Lowest mean warm latency backend: "
            f"`{report.summary.lowest_mean_warm_latency_backend or '-'}`"
        ),
        (
            "- Highest mean throughput backend: "
            f"`{report.summary.highest_mean_throughput_backend or '-'}`"
        ),
        (f"- Max mean abs diff vs torch: `{_format_float(report.summary.max_mean_abs_diff, 8)}`"),
        (
            "- Max cosine distance vs torch: "
            f"`{_format_float(report.summary.max_cosine_distance, 8)}`"
        ),
        "",
        "## Artifacts",
        "",
        f"- Metadata: `{report.metadata_path}`",
        f"- Source checkpoint: `{report.source_checkpoint_path}`",
        f"- ONNX model: `{report.onnx_model_path}`",
        f"- TensorRT report: `{report.tensorrt_report_path}`",
        f"- TensorRT engine: `{report.tensorrt_engine_path}`",
        (f"- ONNX Runtime provider order: `{', '.join(report.onnxruntime_provider_order)}`"),
        "",
        "## Backend Summary",
        "",
        _markdown_table(
            headers=[
                "Backend",
                "Provider",
                "Device",
                "Init (s)",
                "Cold (s)",
                "Warm mean (ms)",
                "Items/s",
                "Frames/s",
                "Peak RSS MiB",
                "Peak GPU MiB",
                "Max diff",
                "Max cosine",
                "Passed",
            ],
            rows=[
                [
                    summary.backend,
                    summary.provider or "-",
                    summary.device,
                    _format_float(summary.mean_initialization_seconds, 6),
                    _format_float(summary.mean_cold_start_seconds, 6),
                    _format_float(summary.mean_warm_latency_ms, 6),
                    _format_float(summary.mean_throughput_items_per_second, 3),
                    _format_float(summary.mean_throughput_frames_per_second, 3),
                    _format_float(summary.peak_process_rss_mib, 3),
                    _format_float(summary.peak_process_gpu_mib, 3),
                    _format_float(summary.max_mean_abs_diff, 8),
                    _format_float(summary.max_cosine_distance, 8),
                    _format_bool(summary.passed),
                ]
                for summary in report.backend_summaries
            ],
        ),
        "",
        "## Workloads",
        "",
        _markdown_table(
            headers=[
                "Backend",
                "Workload",
                "Batch",
                "Frames",
                "Cold (s)",
                "Warm mean (ms)",
                "P95 (ms)",
                "Latency CV",
                "Items/s",
                "RSS dMiB",
                "GPU dMiB",
                "Mean diff",
                "Cosine",
                "Status",
            ],
            rows=[
                [
                    result.backend,
                    result.workload_id,
                    str(result.batch_size),
                    str(result.frame_count),
                    _format_float(result.cold_start_seconds, 6),
                    _format_float(result.warm_mean_latency_ms, 6),
                    _format_float(result.warm_p95_latency_ms, 6),
                    _format_float(result.warm_latency_cv, 6),
                    _format_float(result.throughput_items_per_second, 3),
                    _format_float(result.process_rss_delta_mib, 3),
                    _format_float(result.process_gpu_delta_mib, 3),
                    _format_float(result.mean_abs_diff, 8),
                    _format_float(result.cosine_distance, 8),
                    result.status,
                ]
                for result in _sorted_workload_results(report.workload_results)
            ],
        ),
    ]

    if report.plot_assets:
        lines.extend(["", "## Latency Graphs", ""])
        for asset in report.plot_assets:
            lines.extend(
                [f"### Batch {asset.batch_size}", "", f"![{asset.title}]({asset.path})", ""]
            )

    failing_summaries = [summary for summary in report.backend_summaries if summary.errors]
    if failing_summaries:
        lines.extend(["## Failures", ""])
        for summary in failing_summaries:
            for error in summary.errors:
                lines.append(f"- `{summary.backend}`: {error}")
        lines.append("")

    if report.validation_commands:
        lines.extend(["## Validation Commands", ""])
        lines.extend(f"- `{command}`" for command in report.validation_commands)
        lines.append("")

    if report.notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
        lines.append("")

    lines.extend(
        [
            "## Limits",
            "",
            (
                "- This workflow benchmarks the encoder tensor boundary "
                "`[batch, frames, mel_bins] -> embedding`, not the raw-audio frontend."
            ),
            (
                "- GPU memory values come from process-local `nvidia-smi` snapshots "
                "when available; they may be unavailable on CPU-only hosts."
            ),
            (
                "- Warm latencies include materializing benchmark outputs on the host "
                "side so the three backends are compared on the same embedding handoff "
                "boundary."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_backend_benchmark_report(report: BackendBenchmarkReport) -> WrittenBackendBenchmarkReport:
    output_root = Path(report.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    report_json_path = output_root / BACKEND_BENCHMARK_REPORT_JSON_NAME
    report_markdown_path = output_root / BACKEND_BENCHMARK_REPORT_MARKDOWN_NAME
    workload_rows_path = output_root / BACKEND_BENCHMARK_WORKLOAD_ROWS_NAME

    report_json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_markdown_path.write_text(render_backend_benchmark_markdown(report), encoding="utf-8")
    workload_rows_path.write_text(
        "".join(
            json.dumps(result.to_dict(), sort_keys=True) + "\n"
            for result in report.workload_results
        ),
        encoding="utf-8",
    )

    plot_paths: list[str] = []
    for asset in report.plot_assets:
        plot_path = output_root / asset.path
        plot_path.write_text(
            _render_latency_plot_svg(report=report, asset=asset),
            encoding="utf-8",
        )
        plot_paths.append(str(plot_path))

    source_config_copy_path = None
    if report.source_config_path is not None:
        destination = output_root / "sources" / "backend_benchmark_config.toml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            Path(report.source_config_path).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        source_config_copy_path = str(destination)

    return WrittenBackendBenchmarkReport(
        output_root=str(output_root),
        report_json_path=str(report_json_path),
        report_markdown_path=str(report_markdown_path),
        workload_rows_path=str(workload_rows_path),
        plot_paths=tuple(plot_paths),
        source_config_copy_path=source_config_copy_path,
        summary=report.summary,
    )


def _render_latency_plot_svg(
    *, report: BackendBenchmarkReport, asset: BackendBenchmarkPlotAsset
) -> str:
    points_by_backend: dict[str, list[BackendBenchmarkWorkloadResult]] = {}
    for result in report.workload_results:
        if (
            result.status == "passed"
            and result.batch_size == asset.batch_size
            and result.warm_mean_latency_ms is not None
        ):
            points_by_backend.setdefault(result.backend, []).append(result)

    width = 760
    height = 380
    left = 68
    right = 32
    top = 36
    bottom = 54
    x_values = [
        result.frame_count for results in points_by_backend.values() for result in results
    ] or [0, 1]
    y_values = [
        result.warm_mean_latency_ms
        for results in points_by_backend.values()
        for result in results
        if result.warm_mean_latency_ms is not None
    ] or [0.0, 1.0]
    x_min, x_max = min(x_values), max(x_values)
    y_min = 0.0
    y_max = max(y_values)
    if x_min == x_max:
        x_max += 1
    if y_max <= y_min:
        y_max = y_min + 1.0

    def x_position(value: int) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * (width - left - right)

    def y_position(value: float) -> float:
        return height - bottom - ((value - y_min) / (y_max - y_min)) * (height - top - bottom)

    x_ticks = sorted(set(x_values))
    y_ticks = [y_min + (y_max - y_min) * (index / 4.0) for index in range(5)]
    x_axis_y = height - bottom
    y_axis_mid = (top + height - bottom) / 2
    svg_lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="{left}" y="22" font-family="monospace" font-size="16" '
            f'fill="#111827">{_escape_xml(asset.title)}</text>'
        ),
        (
            f'<line x1="{left}" y1="{x_axis_y}" x2="{width - right}" y2="{x_axis_y}" '
            'stroke="#111827" stroke-width="1.5"/>'
        ),
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{x_axis_y}" '
            'stroke="#111827" stroke-width="1.5"/>'
        ),
    ]

    for tick in x_ticks:
        x = x_position(tick)
        svg_lines.extend(
            [
                (
                    f'<line x1="{x:.2f}" y1="{x_axis_y}" x2="{x:.2f}" '
                    f'y2="{x_axis_y + 6}" stroke="#111827" stroke-width="1"/>'
                ),
                (
                    f'<text x="{x:.2f}" y="{x_axis_y + 22}" text-anchor="middle" '
                    f'font-family="monospace" font-size="12" fill="#374151">{tick}</text>'
                ),
            ]
        )
    for tick in y_ticks:
        y = y_position(tick)
        svg_lines.extend(
            [
                (
                    f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" '
                    'stroke="#111827" stroke-width="1"/>'
                ),
                (
                    f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
                    'stroke="#e5e7eb" stroke-width="1"/>'
                ),
                (
                    f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
                    f'font-family="monospace" font-size="12" fill="#374151">{tick:.2f}</text>'
                ),
            ]
        )

    legend_x = width - right - 180
    legend_y = top + 8
    for index, backend in enumerate(sorted(points_by_backend)):
        color = _BACKEND_COLORS.get(backend, "#6b7280")
        points = sorted(points_by_backend[backend], key=lambda result: result.frame_count)
        polyline = " ".join(
            (
                f"{x_position(result.frame_count):.2f},"
                f"{y_position(result.warm_mean_latency_ms or 0.0):.2f}"
            )
            for result in points
        )
        svg_lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}"/>'
        )
        for result in points:
            svg_lines.append(
                f'<circle cx="{x_position(result.frame_count):.2f}" '
                f'cy="{y_position(result.warm_mean_latency_ms or 0.0):.2f}" '
                f'r="3.5" fill="{color}"/>'
            )
        legend_row_y = legend_y + index * 18
        svg_lines.extend(
            [
                (
                    f'<line x1="{legend_x}" y1="{legend_row_y}" '
                    f'x2="{legend_x + 18}" y2="{legend_row_y}" '
                    f'stroke="{color}" stroke-width="2.5"/>'
                ),
                (
                    f'<text x="{legend_x + 24}" y="{legend_row_y + 4}" '
                    f'font-family="monospace" font-size="12" fill="#111827">'
                    f"{_escape_xml(backend)}</text>"
                ),
            ]
        )

    svg_lines.extend(
        [
            (
                f'<text x="{(left + width - right) / 2:.2f}" y="{height - 14}" '
                'text-anchor="middle" font-family="monospace" font-size="12" '
                'fill="#111827">Frame count</text>'
            ),
            (
                f'<text x="18" y="{y_axis_mid:.2f}" '
                f'transform="rotate(-90 18 {y_axis_mid:.2f})" text-anchor="middle" '
                'font-family="monospace" font-size="12" fill="#111827">'
                "Warm mean latency (ms)</text>"
            ),
            "</svg>",
        ]
    )
    return "\n".join(svg_lines) + "\n"


def _sorted_workload_results(
    results: tuple[BackendBenchmarkWorkloadResult, ...],
) -> list[BackendBenchmarkWorkloadResult]:
    backend_order = {"torch": 0, "onnxruntime": 1, "tensorrt": 2}
    return sorted(
        results,
        key=lambda result: (
            backend_order.get(result.backend, 99),
            result.batch_size,
            result.frame_count,
            result.workload_id,
        ),
    )


def _markdown_table(*, headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_float(value: float | None, digits: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


__all__ = ["render_backend_benchmark_markdown", "write_backend_benchmark_report"]
