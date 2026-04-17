"""Model definitions and inference-facing interfaces."""

from .campp import ArcMarginLoss, CAMPPlusConfig, CAMPPlusEncoder, CosineClassifier
from .eres2netv2 import (
    ERes2NetV2Config,
    ERes2NetV2Encoder,
    load_eres2netv2_encoder_from_checkpoint,
)
from .scoring import (
    average_normalized_embeddings,
    cosine_score_matrix,
    cosine_score_pairs,
    ensure_embedding_matrix,
    l2_normalize_embeddings,
    rank_cosine_scores,
)

__all__ = [
    "ArcMarginLoss",
    "CAMPPlusConfig",
    "CAMPPlusEncoder",
    "CosineClassifier",
    "ERes2NetV2Config",
    "ERes2NetV2Encoder",
    "load_eres2netv2_encoder_from_checkpoint",
    "average_normalized_embeddings",
    "cosine_score_matrix",
    "cosine_score_pairs",
    "ensure_embedding_matrix",
    "l2_normalize_embeddings",
    "rank_cosine_scores",
]
