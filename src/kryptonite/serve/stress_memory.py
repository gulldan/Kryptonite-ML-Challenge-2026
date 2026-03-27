"""Process and CUDA memory helpers for release benchmark workflows."""

from __future__ import annotations

import resource
import sys
from dataclasses import dataclass
from typing import Any

_BYTES_PER_MIB = 1024.0 * 1024.0
_KIB_PER_MIB = 1024.0


@dataclass(frozen=True, slots=True)
class StressMemoryMeasurement:
    """Peak process/CUDA memory observed for one workload."""

    process_peak_rss_mib: float | None
    process_peak_rss_delta_mib: float | None
    cuda_peak_allocated_mib: float | None
    cuda_peak_reserved_mib: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "process_peak_rss_mib": self.process_peak_rss_mib,
            "process_peak_rss_delta_mib": self.process_peak_rss_delta_mib,
            "cuda_peak_allocated_mib": self.cuda_peak_allocated_mib,
            "cuda_peak_reserved_mib": self.cuda_peak_reserved_mib,
        }


def start_memory_measurement() -> float | None:
    """Capture the process RSS baseline and reset CUDA peak counters when possible."""

    _reset_cuda_peak_memory_stats()
    return capture_process_peak_rss_mib()


def finish_memory_measurement(
    baseline_process_peak_rss_mib: float | None,
) -> StressMemoryMeasurement:
    """Capture process/CUDA peaks after a workload."""

    process_peak_rss_mib = capture_process_peak_rss_mib()
    cuda_peak_allocated_mib, cuda_peak_reserved_mib = _capture_cuda_peak_memory_mib()
    process_peak_rss_delta_mib = None
    if baseline_process_peak_rss_mib is not None and process_peak_rss_mib is not None:
        process_peak_rss_delta_mib = round(
            max(process_peak_rss_mib - baseline_process_peak_rss_mib, 0.0),
            3,
        )
    return StressMemoryMeasurement(
        process_peak_rss_mib=process_peak_rss_mib,
        process_peak_rss_delta_mib=process_peak_rss_delta_mib,
        cuda_peak_allocated_mib=cuda_peak_allocated_mib,
        cuda_peak_reserved_mib=cuda_peak_reserved_mib,
    )


def capture_process_peak_rss_mib() -> float | None:
    """Return the current process peak RSS in MiB."""

    try:
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):  # pragma: no cover - defensive for exotic platforms
        return None
    if max_rss <= 0:
        return 0.0
    divisor = _BYTES_PER_MIB if sys.platform == "darwin" else _KIB_PER_MIB
    return round(float(max_rss) / divisor, 3)


def _reset_cuda_peak_memory_stats() -> None:
    torch = _load_torch()
    if torch is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:  # pragma: no cover - backend/runtime dependent
        pass
    try:
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device_index)
    except Exception:  # pragma: no cover - backend/runtime dependent
        return


def _capture_cuda_peak_memory_mib() -> tuple[float | None, float | None]:
    torch = _load_torch()
    if torch is None or not torch.cuda.is_available():
        return None, None
    try:
        torch.cuda.synchronize()
    except Exception:  # pragma: no cover - backend/runtime dependent
        pass
    try:
        allocated = [
            float(torch.cuda.max_memory_allocated(device_index)) / _BYTES_PER_MIB
            for device_index in range(torch.cuda.device_count())
        ]
        reserved = [
            float(torch.cuda.max_memory_reserved(device_index)) / _BYTES_PER_MIB
            for device_index in range(torch.cuda.device_count())
        ]
    except Exception:  # pragma: no cover - backend/runtime dependent
        return None, None
    if not allocated or not reserved:
        return None, None
    return round(max(allocated), 3), round(max(reserved), 3)


def _load_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


__all__ = [
    "StressMemoryMeasurement",
    "capture_process_peak_rss_mib",
    "finish_memory_measurement",
    "start_memory_measurement",
]
