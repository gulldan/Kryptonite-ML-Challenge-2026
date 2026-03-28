"""Public facade for reproducible ONNX Runtime parity reports."""

from .onnx_parity_builder import build_onnx_parity_report
from .onnx_parity_config import (
    ONNXParityArtifactsConfig,
    ONNXParityConfig,
    ONNXParityEvaluationConfig,
    ONNXParityTolerancesConfig,
    ONNXParityVariantConfig,
    load_onnx_parity_config,
)
from .onnx_parity_models import (
    ONNX_PARITY_AUDIO_ROWS_NAME,
    ONNX_PARITY_REPORT_JSON_NAME,
    ONNX_PARITY_REPORT_MARKDOWN_NAME,
    ONNX_PARITY_TRIAL_ROWS_NAME,
    ONNXParityAudioRecord,
    ONNXParityPromotionState,
    ONNXParityReport,
    ONNXParitySummary,
    ONNXParityTrialRecord,
    ONNXParityVariantSummary,
    WrittenONNXParityReport,
)
from .onnx_parity_rendering import render_onnx_parity_markdown, write_onnx_parity_report

__all__ = [
    "ONNX_PARITY_AUDIO_ROWS_NAME",
    "ONNX_PARITY_REPORT_JSON_NAME",
    "ONNX_PARITY_REPORT_MARKDOWN_NAME",
    "ONNX_PARITY_TRIAL_ROWS_NAME",
    "ONNXParityArtifactsConfig",
    "ONNXParityAudioRecord",
    "ONNXParityConfig",
    "ONNXParityEvaluationConfig",
    "ONNXParityPromotionState",
    "ONNXParityReport",
    "ONNXParitySummary",
    "ONNXParityTolerancesConfig",
    "ONNXParityTrialRecord",
    "ONNXParityVariantConfig",
    "ONNXParityVariantSummary",
    "WrittenONNXParityReport",
    "build_onnx_parity_report",
    "load_onnx_parity_config",
    "render_onnx_parity_markdown",
    "write_onnx_parity_report",
]
