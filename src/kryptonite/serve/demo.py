"""Helpers for the lightweight browser demo built on top of the shared inferencer."""

from __future__ import annotations

import base64
import binascii
import json
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kryptonite.deployment import resolve_project_path
from kryptonite.eval.verification_threshold_calibration import (
    VERIFICATION_THRESHOLD_CALIBRATION_JSON_NAME,
)

from .api_models import (
    DemoAudioUpload,
    DemoCompareRequest,
    DemoEnrollmentRequest,
    DemoVerifyRequest,
)

if TYPE_CHECKING:
    from kryptonite.config import ProjectConfig

    from .inferencer import Inferencer


_FALLBACK_DEMO_THRESHOLD = 0.995
_VERIFICATION_EVAL_REPORT_NAME = "verification_eval_report.json"


@dataclass(frozen=True, slots=True)
class DemoThresholdReference:
    value: float
    source: str
    profile: str | None = None
    origin_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "value": round(float(self.value), 6),
            "source": self.source,
        }
        if self.profile is not None:
            payload["profile"] = self.profile
        if self.origin_path is not None:
            payload["origin_path"] = self.origin_path
        return payload

    def override(self, value: float) -> DemoThresholdReference:
        return DemoThresholdReference(
            value=float(value),
            source="request_override",
            profile=self.profile,
            origin_path=self.origin_path,
        )


def build_demo_state(
    *,
    inferencer: Inferencer,
    config: ProjectConfig,
    threshold: DemoThresholdReference,
) -> dict[str, object]:
    health = inferencer.health_payload()
    enrollments = inferencer.list_enrollments()
    return {
        "service": {
            "status": health["status"],
            "selected_backend": health["selected_backend"],
            "model_bundle": dict(health["model_bundle"]),
            "inferencer": dict(health["inferencer"]),
            "runtime": dict(health["runtime"]),
            "artifacts": {
                "passed": health["artifacts"]["passed"],
                "scope": health["artifacts"]["scope"],
            },
            "runtime_store": dict(health["enrollment_cache"]["runtime_store"]),
        },
        "threshold": threshold.to_dict(),
        "default_stage": "demo",
        "supported_formats": [".wav", ".flac", ".mp3"],
        "deployment": {
            "demo_subset_root": config.deployment.demo_subset_root,
            "enrollment_cache_root": config.deployment.enrollment_cache_root,
        },
        **enrollments,
    }


def resolve_demo_threshold(*, config: ProjectConfig) -> DemoThresholdReference:
    project_root = resolve_project_path(config.paths.project_root, ".")
    artifacts_root = resolve_project_path(config.paths.project_root, config.paths.artifacts_root)
    calibration_threshold = _resolve_threshold_from_calibration(
        artifacts_root,
        project_root=project_root,
    )
    if calibration_threshold is not None:
        return calibration_threshold

    eval_threshold = _resolve_threshold_from_eval_report(
        artifacts_root,
        project_root=project_root,
    )
    if eval_threshold is not None:
        return eval_threshold

    return DemoThresholdReference(
        value=_FALLBACK_DEMO_THRESHOLD,
        source="builtin_default",
        profile="demo",
    )


def run_demo_compare(
    *,
    inferencer: Inferencer,
    request: DemoCompareRequest,
    default_threshold: DemoThresholdReference,
) -> dict[str, object]:
    threshold = (
        default_threshold
        if request.threshold is None
        else default_threshold.override(request.threshold)
    )
    with _materialize_audio_uploads([request.left_audio, request.right_audio]) as audio_paths:
        started = time.perf_counter()
        embed_payload = inferencer.embed_audio_paths(
            audio_paths=audio_paths,
            stage=request.stage,
            include_embeddings=True,
        )
        score_payload = inferencer.score_pairwise(
            left=embed_payload["items"][0]["embedding"],
            right=embed_payload["items"][1]["embedding"],
            normalize=request.normalize,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

    score = float(score_payload["scores"][0])
    return {
        "mode": "demo_compare",
        "stage": embed_payload["stage"],
        "normalized": bool(score_payload["normalized"]),
        "score": round(score, 8),
        "decision": bool(score >= threshold.value),
        "threshold": threshold.to_dict(),
        "latency_ms": round(latency_ms, 3),
        "backend": dict(embed_payload["backend"]),
        "left_audio": _strip_embedding(embed_payload["items"][0]),
        "right_audio": _strip_embedding(embed_payload["items"][1]),
    }


def run_demo_enroll(
    *,
    inferencer: Inferencer,
    request: DemoEnrollmentRequest,
) -> dict[str, object]:
    with _materialize_audio_uploads(request.audio_files) as audio_paths:
        started = time.perf_counter()
        payload = inferencer.enroll_audio_paths(
            enrollment_id=request.enrollment_id,
            audio_paths=audio_paths,
            stage=request.stage,
            metadata=request.metadata,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

    return {
        "mode": "demo_enroll",
        "stage": payload["stage"],
        "enrollment_id": payload["enrollment_id"],
        "sample_count": payload["sample_count"],
        "embedding_dim": payload["embedding_dim"],
        "metadata": payload["metadata"],
        "replaced": bool(payload["replaced"]),
        "latency_ms": round(latency_ms, 3),
        "backend": dict(payload["backend"]),
        "audio_items": list(payload["audio_items"]),
    }


def run_demo_verify(
    *,
    inferencer: Inferencer,
    request: DemoVerifyRequest,
    default_threshold: DemoThresholdReference,
) -> dict[str, object]:
    threshold = (
        default_threshold
        if request.threshold is None
        else default_threshold.override(request.threshold)
    )
    with _materialize_audio_uploads([request.audio_file]) as audio_paths:
        started = time.perf_counter()
        payload = inferencer.verify_audio_paths(
            enrollment_id=request.enrollment_id,
            audio_paths=audio_paths,
            stage=request.stage,
            normalize=request.normalize,
            threshold=threshold.value,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

    return {
        "mode": "demo_verify",
        "stage": payload["stage"],
        "normalized": bool(payload["normalized"]),
        "enrollment_id": payload["enrollment_id"],
        "score": round(float(payload["scores"][0]), 8),
        "decision": bool(payload["decisions"][0]),
        "threshold": threshold.to_dict(),
        "latency_ms": round(latency_ms, 3),
        "backend": dict(payload["backend"]),
        "probe_audio": dict(payload["probe_items"][0]),
    }


def _resolve_threshold_from_calibration(
    artifacts_root: Path,
    *,
    project_root: Path,
) -> DemoThresholdReference | None:
    for path in _iter_latest_artifacts(
        artifacts_root, VERIFICATION_THRESHOLD_CALIBRATION_JSON_NAME
    ):
        payload = _load_json(path)
        global_profiles = payload.get("global_profiles")
        if not isinstance(global_profiles, list):
            continue
        for candidate in global_profiles:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("name") != "demo":
                continue
            threshold = candidate.get("threshold")
            if not isinstance(threshold, int | float):
                continue
            return DemoThresholdReference(
                value=float(threshold),
                source="calibration_profile",
                profile="demo",
                origin_path=_display_path(path, project_root),
            )
    return None


def _resolve_threshold_from_eval_report(
    artifacts_root: Path,
    *,
    project_root: Path,
) -> DemoThresholdReference | None:
    for path in _iter_latest_artifacts(artifacts_root, _VERIFICATION_EVAL_REPORT_NAME):
        payload = _load_json(path)
        summary = payload.get("error_analysis", {}).get("summary")
        if not isinstance(summary, dict):
            continue
        threshold = summary.get("decision_threshold")
        if not isinstance(threshold, int | float):
            continue
        threshold_source = summary.get("threshold_source")
        resolved_source = (
            str(threshold_source).strip() if isinstance(threshold_source, str) else "unknown"
        )
        return DemoThresholdReference(
            value=float(threshold),
            source=f"verification_eval:{resolved_source}",
            profile="balanced",
            origin_path=_display_path(path, project_root),
        )
    return None


def _iter_latest_artifacts(root: Path, filename: str) -> list[Path]:
    return sorted(
        root.rglob(filename),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _strip_embedding(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "embedding"}


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


@contextmanager
def _materialize_audio_uploads(
    uploads: list[DemoAudioUpload],
) -> Iterator[list[str]]:
    with tempfile.TemporaryDirectory(prefix="kryptonite-demo-") as tmp_dir:
        root = Path(tmp_dir)
        audio_paths: list[str] = []
        for index, upload in enumerate(uploads):
            suffix = Path(upload.filename).suffix.lower()
            target = root / f"upload-{index:02d}{suffix}"
            try:
                payload = base64.b64decode(upload.content_base64, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError(
                    f"Upload {upload.filename!r} is not valid base64 audio content."
                ) from exc
            if not payload:
                raise ValueError(f"Upload {upload.filename!r} is empty.")
            target.write_bytes(payload)
            audio_paths.append(str(target))
        yield audio_paths


__all__ = [
    "DemoThresholdReference",
    "build_demo_state",
    "resolve_demo_threshold",
    "run_demo_compare",
    "run_demo_enroll",
    "run_demo_verify",
]
