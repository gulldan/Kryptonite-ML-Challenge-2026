"""Embedding-atlas projection and interactive report generation."""

from .models import (
    ALLOWED_PROJECTION_METHODS,
    POINTS_JSONL_NAME,
    REPORT_HTML_NAME,
    REPORT_JSON_NAME,
    REPORT_MARKDOWN_NAME,
    EmbeddingAtlasPoint,
    EmbeddingAtlasReport,
    EmbeddingAtlasSummary,
    NeighborPoint,
    ProjectionMethod,
    WrittenEmbeddingAtlasArtifacts,
)
from .reporting import (
    build_embedding_atlas,
    render_embedding_atlas_html,
    render_embedding_atlas_markdown,
    write_embedding_atlas_report,
)

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
    "build_embedding_atlas",
    "render_embedding_atlas_html",
    "render_embedding_atlas_markdown",
    "write_embedding_atlas_report",
]
