"""CAM++ speaker-verification model primitives."""

from .losses import ArcMarginLoss, CosineClassifier
from .model import CAMPPlusConfig, CAMPPlusEncoder

__all__ = [
    "ArcMarginLoss",
    "CAMPPlusConfig",
    "CAMPPlusEncoder",
    "CosineClassifier",
]
