"""Model definitions and inference-facing interfaces."""

from .campp import ArcMarginLoss, CAMPPlusConfig, CAMPPlusEncoder, CosineClassifier
from .eres2netv2 import ERes2NetV2Config, ERes2NetV2Encoder

__all__ = [
    "ArcMarginLoss",
    "CAMPPlusConfig",
    "CAMPPlusEncoder",
    "CosineClassifier",
    "ERes2NetV2Config",
    "ERes2NetV2Encoder",
]
