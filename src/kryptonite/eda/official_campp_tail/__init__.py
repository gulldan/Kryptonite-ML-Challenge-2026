"""Official CAM++ tail pipeline exposed as reusable src modules."""

from .config import OfficialCamPPTailConfig, parse_args
from .extraction import (
    FrontendCacheStats,
    _try_extract_embeddings_from_contiguous_frontend_pack,
    load_or_extract_embeddings,
)
from .pipeline import main, run_official_campp_tail

__all__ = [
    "FrontendCacheStats",
    "OfficialCamPPTailConfig",
    "load_or_extract_embeddings",
    "main",
    "parse_args",
    "run_official_campp_tail",
    "_try_extract_embeddings_from_contiguous_frontend_pack",
]
