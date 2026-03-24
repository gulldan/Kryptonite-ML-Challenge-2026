"""Model definitions and inference-facing interfaces."""

from .campp import ArcMarginLoss, CAMPPlusConfig, CAMPPlusEncoder, CosineClassifier

__all__ = [
    "ArcMarginLoss",
    "CAMPPlusConfig",
    "CAMPPlusEncoder",
    "CosineClassifier",
]
