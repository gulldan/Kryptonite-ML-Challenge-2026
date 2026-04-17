"""Build and write identification evaluation reports (JSON + Markdown)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .identification_metrics import (
    IdentificationMetricsSummary,
    compute_identification_metrics,
)

IDENTIFICATION_REPORT_JSON_NAME = "identification_eval_report.json"
IDENTIFICATION_REPORT_MARKDOWN_NAME = "identification_eval_report.md"
IDENTIFICATION_CMC_CURVE_JSONL_NAME = "identification_cmc_curve.jsonl"
IDENTIFICATION_OPERATING_POINTS_JSONL_NAME = "identification_operating_points.jsonl"


@dataclass(frozen=True, slots=True)
class IdentificationReportInputs:
    embeddings_path: str
    metadata_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IdentificationEvaluationReport:
    inputs: IdentificationReportInputs
    summary: IdentificationMetricsSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": self.inputs.to_dict(),
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class WrittenIdentificationEvaluationReport:
    report_json_path: str
    report_markdown_path: str
    cmc_curve_path: str
    operating_points_path: str
    summary: IdentificationMetricsSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "cmc_curve_path": self.cmc_curve_path,
            "operating_points_path": self.operating_points_path,
            "summary": self.summary.to_dict(),
        }


def build_identification_evaluation_report(
    *,
    embeddings_path: str,
    metadata_rows: list[dict[str, Any]],
    metadata_path: str | None = None,
    max_rank: int | None = None,
    gallery_fraction: float = 0.1,
    seed: int = 42,
) -> IdentificationEvaluationReport | None:
    """Build identification report from embeddings and metadata.

    If metadata contains explicit ``role`` fields (enrollment/test), uses those.
    Otherwise, auto-splits each speaker's utterances: a fraction goes to the
    gallery (enrollment) and the rest become queries (probes).

    Returns None if the split yields empty query or gallery sets.
    """
    query_indices, query_speaker_ids, gallery_indices, gallery_speaker_ids = _split_query_gallery(
        metadata_rows, gallery_fraction=gallery_fraction, seed=seed
    )

    if not query_indices or not gallery_indices:
        return None

    npz = np.load(embeddings_path)
    all_embeddings = np.asarray(npz["embeddings"], dtype=np.float64)

    query_embeddings = all_embeddings[query_indices]
    gallery_embeddings = all_embeddings[gallery_indices]

    metrics = compute_identification_metrics(
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        query_speaker_ids=query_speaker_ids,
        gallery_speaker_ids=gallery_speaker_ids,
        max_rank=max_rank,
    )

    return IdentificationEvaluationReport(
        inputs=IdentificationReportInputs(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
        ),
        summary=metrics,
    )


def render_identification_evaluation_markdown(
    report: IdentificationEvaluationReport,
) -> str:
    """Render identification evaluation results as Markdown."""
    s = report.summary
    lines = [
        "# Identification Evaluation Report",
        "",
        "## Inputs",
        "",
        f"- Embeddings: `{report.inputs.embeddings_path}`",
    ]
    if report.inputs.metadata_path:
        lines.append(f"- Metadata: `{report.inputs.metadata_path}`")
    lines.extend(
        [
            "",
            "## Gallery / Query Split",
            "",
            f"- Query (probe) count: `{s.query_count}`",
            f"- Gallery (enrollment) count: `{s.gallery_count}`",
            f"- Gallery speakers: `{s.gallery_speaker_count}`",
            "",
            "## Rank-K Identification Accuracy (CMC)",
            "",
            f"- Rank-1: `{s.rank_1_accuracy:.4f}`",
            f"- Rank-5: `{s.rank_5_accuracy:.4f}`",
            f"- Rank-10: `{s.rank_10_accuracy:.4f}`",
            f"- Rank-20: `{s.rank_20_accuracy:.4f}`",
            "",
            "## Open-Set Metrics (FNIR / FPIR)",
            "",
        ]
    )
    if s.fnir_at_fpir_0_01 is not None:
        lines.append(f"- FNIR @ FPIR=0.01: `{s.fnir_at_fpir_0_01:.4f}`")
    if s.fnir_at_fpir_0_1 is not None:
        lines.append(f"- FNIR @ FPIR=0.1: `{s.fnir_at_fpir_0_1:.4f}`")
    lines.append("")
    return "\n".join(lines)


def write_identification_evaluation_report(
    report: IdentificationEvaluationReport,
    *,
    output_root: Path,
) -> WrittenIdentificationEvaluationReport:
    """Write identification report artifacts to disk."""
    json_path = output_root / IDENTIFICATION_REPORT_JSON_NAME
    md_path = output_root / IDENTIFICATION_REPORT_MARKDOWN_NAME
    cmc_path = output_root / IDENTIFICATION_CMC_CURVE_JSONL_NAME
    ops_path = output_root / IDENTIFICATION_OPERATING_POINTS_JSONL_NAME

    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_identification_evaluation_markdown(report), encoding="utf-8")
    cmc_path.write_text(
        "".join(json.dumps(p.to_dict(), sort_keys=True) + "\n" for p in report.summary.cmc_curve),
        encoding="utf-8",
    )
    ops_path.write_text(
        "".join(
            json.dumps(p.to_dict(), sort_keys=True) + "\n" for p in report.summary.operating_points
        ),
        encoding="utf-8",
    )

    return WrittenIdentificationEvaluationReport(
        report_json_path=str(json_path),
        report_markdown_path=str(md_path),
        cmc_curve_path=str(cmc_path),
        operating_points_path=str(ops_path),
        summary=report.summary,
    )


def _split_query_gallery(
    metadata_rows: list[dict[str, Any]],
    *,
    gallery_fraction: float,
    seed: int,
) -> tuple[list[int], list[str], list[int], list[str]]:
    """Split metadata into query and gallery sets.

    If roles are explicit, use them directly.  Otherwise, for each speaker,
    deterministically assign ``gallery_fraction`` of utterances to the gallery
    (at least 1) and the rest to the query set.
    """
    query_indices: list[int] = []
    query_speaker_ids: list[str] = []
    gallery_indices: list[int] = []
    gallery_speaker_ids: list[str] = []

    # Try explicit roles first
    for idx, row in enumerate(metadata_rows):
        role = row.get("role") or ""
        speaker_id = row.get("speaker_id", "")
        if not speaker_id:
            continue
        if role == "test":
            query_indices.append(idx)
            query_speaker_ids.append(speaker_id)
        elif role == "enrollment":
            gallery_indices.append(idx)
            gallery_speaker_ids.append(speaker_id)

    if query_indices and gallery_indices:
        return query_indices, query_speaker_ids, gallery_indices, gallery_speaker_ids

    # Auto-split: group by speaker, assign gallery_fraction as gallery
    from collections import defaultdict

    speaker_rows: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(metadata_rows):
        speaker_id = row.get("speaker_id", "")
        if speaker_id:
            speaker_rows[speaker_id].append(idx)

    if not speaker_rows:
        return [], [], [], []

    rng = np.random.default_rng(seed)
    query_indices = []
    query_speaker_ids = []
    gallery_indices = []
    gallery_speaker_ids = []

    for speaker_id in sorted(speaker_rows):
        indices = speaker_rows[speaker_id]
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_gallery = max(1, int(len(shuffled) * gallery_fraction))
        for i, idx in enumerate(shuffled):
            if i < n_gallery:
                gallery_indices.append(idx)
                gallery_speaker_ids.append(speaker_id)
            else:
                query_indices.append(idx)
                query_speaker_ids.append(speaker_id)

    return query_indices, query_speaker_ids, gallery_indices, gallery_speaker_ids


__all__ = [
    "IDENTIFICATION_CMC_CURVE_JSONL_NAME",
    "IDENTIFICATION_OPERATING_POINTS_JSONL_NAME",
    "IDENTIFICATION_REPORT_JSON_NAME",
    "IDENTIFICATION_REPORT_MARKDOWN_NAME",
    "IdentificationEvaluationReport",
    "IdentificationReportInputs",
    "WrittenIdentificationEvaluationReport",
    "build_identification_evaluation_report",
    "render_identification_evaluation_markdown",
    "write_identification_evaluation_report",
]
