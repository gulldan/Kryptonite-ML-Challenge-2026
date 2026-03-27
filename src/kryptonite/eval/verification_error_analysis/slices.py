"""Slice derivation helpers for verification error analysis.

This module intentionally reuses the shared verification slice logic so the main
report and the thresholded error-analysis view stay aligned when new slice types
are introduced.
"""

from __future__ import annotations

from typing import Any

from ..verification_slices import derive_slice_value as _derive_slice_value


def derive_slice_value(
    field_name: str,
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> str | None:
    return _derive_slice_value(
        field_name,
        left_metadata=left_metadata,
        right_metadata=right_metadata,
    )


__all__ = ["derive_slice_value"]
