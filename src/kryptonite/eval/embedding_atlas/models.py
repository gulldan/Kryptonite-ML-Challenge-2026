"""Datamodels for embedding-atlas projection and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProjectionMethod = Literal["pca", "cosine_pca"]

ALLOWED_PROJECTION_METHODS: tuple[ProjectionMethod, ...] = ("pca", "cosine_pca")
POINTS_JSONL_NAME = "embedding_atlas_points.jsonl"
REPORT_JSON_NAME = "embedding_atlas_report.json"
REPORT_MARKDOWN_NAME = "embedding_atlas_report.md"
REPORT_HTML_NAME = "embedding_atlas.html"


@dataclass(frozen=True, slots=True)
class NeighborPoint:
    point_id: str
    label: str
    distance: float

    def to_dict(self) -> dict[str, object]:
        return {
            "point_id": self.point_id,
            "label": self.label,
            "distance": self.distance,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingAtlasPoint:
    point_index: int
    point_id: str
    label: str
    color_key: str
    x: float
    y: float
    metadata: dict[str, str]
    search_text: str
    audio_href: str | None = None
    image_href: str | None = None
    neighbors: tuple[NeighborPoint, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "point_id": self.point_id,
            "label": self.label,
            "color_key": self.color_key,
            "x": self.x,
            "y": self.y,
            "metadata": dict(self.metadata),
            "search_text": self.search_text,
            "audio_href": self.audio_href,
            "image_href": self.image_href,
            "neighbors": [neighbor.to_dict() for neighbor in self.neighbors],
        }


@dataclass(frozen=True, slots=True)
class EmbeddingAtlasSummary:
    point_count: int
    embedding_dim: int
    projection_method: ProjectionMethod
    point_id_field: str
    label_field: str
    color_by_field: str
    search_fields: tuple[str, ...]
    neighbor_count: int
    explained_variance_ratio_2d: float
    distinct_label_count: int
    distinct_color_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "point_count": self.point_count,
            "embedding_dim": self.embedding_dim,
            "projection_method": self.projection_method,
            "point_id_field": self.point_id_field,
            "label_field": self.label_field,
            "color_by_field": self.color_by_field,
            "search_fields": list(self.search_fields),
            "neighbor_count": self.neighbor_count,
            "explained_variance_ratio_2d": self.explained_variance_ratio_2d,
            "distinct_label_count": self.distinct_label_count,
            "distinct_color_count": self.distinct_color_count,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingAtlasReport:
    generated_at: str
    project_root: str
    output_root: str
    title: str
    embeddings_path: str
    metadata_path: str
    summary: EmbeddingAtlasSummary
    points: tuple[EmbeddingAtlasPoint, ...]

    def to_dict(self, *, include_points: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "generated_at": self.generated_at,
            "project_root": self.project_root,
            "output_root": self.output_root,
            "title": self.title,
            "embeddings_path": self.embeddings_path,
            "metadata_path": self.metadata_path,
            "summary": self.summary.to_dict(),
        }
        if include_points:
            payload["points"] = [point.to_dict() for point in self.points]
        return payload


@dataclass(frozen=True, slots=True)
class WrittenEmbeddingAtlasArtifacts:
    output_root: str
    points_path: str
    json_path: str
    markdown_path: str
    html_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_root": self.output_root,
            "points_path": self.points_path,
            "json_path": self.json_path,
            "markdown_path": self.markdown_path,
            "html_path": self.html_path,
        }


__all__ = [
    "ALLOWED_PROJECTION_METHODS",
    "POINTS_JSONL_NAME",
    "REPORT_HTML_NAME",
    "REPORT_JSON_NAME",
    "REPORT_MARKDOWN_NAME",
    "EmbeddingAtlasPoint",
    "EmbeddingAtlasReport",
    "EmbeddingAtlasSummary",
    "NeighborPoint",
    "ProjectionMethod",
    "WrittenEmbeddingAtlasArtifacts",
]
