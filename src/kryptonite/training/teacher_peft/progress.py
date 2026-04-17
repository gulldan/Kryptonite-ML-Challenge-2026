"""Progress logging helpers for teacher PEFT runs."""

from __future__ import annotations

import math
import sys

import torch


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def format_eta(*, elapsed_seconds: float, completed_items: int, total_items: int | None) -> str:
    if total_items is None or completed_items <= 0 or completed_items >= total_items:
        return "done" if total_items is not None and completed_items >= total_items else "n/a"
    seconds_per_item = elapsed_seconds / float(completed_items)
    return format_duration(seconds_per_item * float(total_items - completed_items))


def resolve_log_interval(
    total_items: int | None,
    *,
    target_updates: int = 20,
    min_interval: int = 10,
    max_interval: int = 250,
) -> int:
    if total_items is None or total_items <= 0:
        return min_interval
    return max(min_interval, min(max_interval, math.ceil(total_items / float(target_updates))))


def format_cuda_memory(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "cpu"
    allocated_gib = torch.cuda.memory_allocated(device) / float(1024**3)
    reserved_gib = torch.cuda.memory_reserved(device) / float(1024**3)
    return f"{allocated_gib:.2f}/{reserved_gib:.2f}GiB"


def emit_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
