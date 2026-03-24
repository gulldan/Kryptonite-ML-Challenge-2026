"""Embedding-atlas projection and interactive report generation."""

from .export import (
    EMBEDDINGS_NPZ_NAME,
    EXPORT_REPORT_JSON_NAME,
    METADATA_JSONL_NAME,
    METADATA_PARQUET_NAME,
    SUPPORTED_BASELINE_EMBEDDING_MODES,
    SUPPORTED_METADATA_SOURCE_FORMATS,
    ManifestEmbeddingExportArtifacts,
    ManifestEmbeddingExportSummary,
    export_manifest_fbank_embeddings,
)
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
    "EMBEDDINGS_NPZ_NAME",
    "EXPORT_REPORT_JSON_NAME",
    "METADATA_JSONL_NAME",
    "METADATA_PARQUET_NAME",
    "POINTS_JSONL_NAME",
    "REPORT_HTML_NAME",
    "REPORT_JSON_NAME",
    "REPORT_MARKDOWN_NAME",
    "EmbeddingAtlasPoint",
    "EmbeddingAtlasReport",
    "EmbeddingAtlasSummary",
    "ManifestEmbeddingExportArtifacts",
    "ManifestEmbeddingExportSummary",
    "NeighborPoint",
    "ProjectionMethod",
    "SUPPORTED_BASELINE_EMBEDDING_MODES",
    "SUPPORTED_METADATA_SOURCE_FORMATS",
    "WrittenEmbeddingAtlasArtifacts",
    "build_embedding_atlas",
    "export_manifest_fbank_embeddings",
    "render_embedding_atlas_html",
    "render_embedding_atlas_markdown",
    "write_embedding_atlas_report",
]
