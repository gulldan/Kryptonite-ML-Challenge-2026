#!/usr/bin/env python3
"""Download and materialize organizer-facing model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "deployment" / "artifacts.toml"


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    artifact_id: str
    source: str
    filename: str
    public_key: str | None = None
    file_id: str | None = None
    archive_type: str = ""
    archive_sha256: str = ""
    archive_member: str = ""
    archive_prefix: str = ""
    dest_path: str = ""
    dest_dir: str = ""
    target_sha256: str = ""
    required_files: tuple[str, ...] = ()
    url_env: str = ""

    @property
    def cache_path(self) -> Path:
        return PROJECT_ROOT / "data" / ".artifact_cache" / self.filename

    @property
    def resolved_dest_path(self) -> Path | None:
        return None if not self.dest_path else (PROJECT_ROOT / self.dest_path)

    @property
    def resolved_dest_dir(self) -> Path | None:
        return None if not self.dest_dir else (PROJECT_ROOT / self.dest_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("campp-pt", "w2v-trt"))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--stage-dir",
        default="",
        help=(
            "Optional directory used to stage only the selected model artifacts for "
            "Docker image assembly. The staged tree preserves repo-relative paths."
        ),
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    models, artifacts = load_manifest(manifest_path)
    specs = model_artifact_specs(models=models, artifacts=artifacts, model=args.model)
    rows = ensure_model_artifacts(specs=specs, offline=args.offline, force=args.force)
    staged_rows: list[dict[str, str]] = []
    if args.stage_dir:
        staged_rows = stage_model_artifacts(specs=specs, stage_dir=Path(args.stage_dir))
    if args.as_json:
        payload: dict[str, Any] = {"artifacts": rows}
        if staged_rows:
            payload["staged_artifacts"] = staged_rows
            payload["stage_dir"] = str(Path(args.stage_dir))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for state in rows:
        print(
            "[artifact] "
            f"id={state['artifact_id']} status={state['status']} "
            f"destination={state['destination']}",
            flush=True,
        )
    if staged_rows:
        print(
            f"[artifact-stage] count={len(staged_rows)} stage_dir={Path(args.stage_dir)}",
            flush=True,
        )


def model_artifact_specs(
    *,
    models: dict[str, list[str]],
    artifacts: dict[str, ArtifactSpec],
    model: str,
) -> list[ArtifactSpec]:
    artifact_ids = models[model]
    return [artifacts[artifact_id] for artifact_id in artifact_ids]


def ensure_model_artifacts(
    *,
    specs: list[ArtifactSpec],
    offline: bool,
    force: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in specs:
        state = ensure_artifact(spec, offline=offline, force=force)
        rows.append(state)
    return rows


def load_manifest(path: Path) -> tuple[dict[str, list[str]], dict[str, ArtifactSpec]]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    models_raw = payload.get("models")
    artifacts_raw = payload.get("artifacts")
    if not isinstance(models_raw, dict) or not isinstance(artifacts_raw, dict):
        raise ValueError(f"Manifest must define [models] and [artifacts.*]: {path}")

    models: dict[str, list[str]] = {}
    for model_name, artifact_ids in models_raw.items():
        if not isinstance(model_name, str) or not isinstance(artifact_ids, list):
            raise ValueError(f"Invalid model entry in {path}: {model_name!r}")
        models[model_name] = [str(item) for item in artifact_ids]

    artifacts: dict[str, ArtifactSpec] = {}
    for artifact_id, raw in artifacts_raw.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Artifact {artifact_id!r} must be a table.")
        artifacts[artifact_id] = ArtifactSpec(
            artifact_id=str(artifact_id),
            source=str(raw.get("source", "")),
            filename=str(raw.get("filename", "")),
            public_key=_as_optional_str(raw.get("public_key")),
            file_id=_as_optional_str(raw.get("file_id")),
            archive_type=str(raw.get("archive_type", "")),
            archive_sha256=str(raw.get("archive_sha256", "")),
            archive_member=str(raw.get("archive_member", "")),
            archive_prefix=str(raw.get("archive_prefix", "")),
            dest_path=str(raw.get("dest_path", "")),
            dest_dir=str(raw.get("dest_dir", "")),
            target_sha256=str(raw.get("target_sha256", "")),
            required_files=tuple(str(item) for item in raw.get("required_files", []) or []),
            url_env=str(raw.get("url_env", "")),
        )
    return models, artifacts


def ensure_artifact(spec: ArtifactSpec, *, offline: bool, force: bool) -> dict[str, str]:
    destination = spec.resolved_dest_path or spec.resolved_dest_dir
    if destination is None:
        raise ValueError(f"Artifact {spec.artifact_id} does not define dest_path or dest_dir.")

    if not force and destination_ready(spec):
        return {
            "artifact_id": spec.artifact_id,
            "status": "ready",
            "destination": str(destination),
        }

    if offline:
        raise RuntimeError(
            f"Artifact {spec.artifact_id} is missing or invalid and --offline is set: {destination}"
        )

    spec.cache_path.parent.mkdir(parents=True, exist_ok=True)
    download_if_needed(spec, force=force)
    materialize(spec)
    if not destination_ready(spec):
        raise RuntimeError(
            f"Artifact {spec.artifact_id} failed verification after materialization."
        )
    return {
        "artifact_id": spec.artifact_id,
        "status": "downloaded",
        "destination": str(destination),
    }


def stage_model_artifacts(
    *,
    specs: list[ArtifactSpec],
    stage_dir: Path,
) -> list[dict[str, str]]:
    stage_dir = stage_dir.resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)
    for child in stage_dir.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    rows: list[dict[str, str]] = []
    for spec in specs:
        source = spec.resolved_dest_path or spec.resolved_dest_dir
        if source is None:
            raise ValueError(f"Artifact {spec.artifact_id} has no resolved destination.")
        if not destination_ready(spec):
            raise RuntimeError(
                f"Artifact {spec.artifact_id} is missing or invalid and cannot be staged: {source}"
            )
        relative_path = source.relative_to(PROJECT_ROOT)
        target = stage_dir / relative_path
        _copy_stage_source(source=source, target=target)
        rows.append(
            {
                "artifact_id": spec.artifact_id,
                "source": str(source),
                "staged_path": str(target),
            }
        )
        if source.is_file():
            metadata_path = tensorrt_engine_metadata_path(source)
            if metadata_path.is_file():
                metadata_target = tensorrt_engine_metadata_path(target)
                metadata_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(metadata_path, metadata_target)
                rows.append(
                    {
                        "artifact_id": f"{spec.artifact_id}.metadata",
                        "source": str(metadata_path),
                        "staged_path": str(metadata_target),
                    }
                )
    manifest_path = stage_dir / "staged_artifacts.json"
    manifest_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (stage_dir / ".gitkeep").write_text("", encoding="utf-8")
    return rows


def _copy_stage_source(*, source: Path, target: Path) -> None:
    if source.is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def tensorrt_engine_metadata_path(engine_path: Path) -> Path:
    return engine_path.with_name(f"{engine_path.name}.metadata.json")


def destination_ready(spec: ArtifactSpec) -> bool:
    dest_path = spec.resolved_dest_path
    if dest_path is not None:
        if not dest_path.is_file():
            return False
        if spec.target_sha256:
            return sha256_file(dest_path) == spec.target_sha256
        return True

    dest_dir = spec.resolved_dest_dir
    assert dest_dir is not None
    if not dest_dir.is_dir():
        return False
    for relative in spec.required_files:
        if not (dest_dir / relative).is_file():
            return False
    return True


def download_if_needed(spec: ArtifactSpec, *, force: bool) -> None:
    if spec.cache_path.is_file() and not force:
        if not spec.archive_sha256 or sha256_file(spec.cache_path) == spec.archive_sha256:
            return
    download_url = resolve_download_url(spec)
    with requests.Session() as session:
        session.headers["User-Agent"] = "kryptonite-submit-artifacts/1.0"
        response = session.get(download_url, stream=True, timeout=60)
        response.raise_for_status()
        tmp_path = spec.cache_path.with_suffix(spec.cache_path.suffix + ".part")
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        tmp_path.replace(spec.cache_path)
    if spec.archive_sha256:
        digest = sha256_file(spec.cache_path)
        if digest != spec.archive_sha256:
            spec.cache_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {spec.artifact_id}: "
                f"expected {spec.archive_sha256}, got {digest}"
            )


def resolve_download_url(spec: ArtifactSpec) -> str:
    env_override = os.environ.get(spec.url_env) if spec.url_env else None
    if env_override:
        return env_override
    if spec.source == "yandex_public":
        if not spec.public_key:
            raise ValueError(f"Artifact {spec.artifact_id} is missing public_key.")
        public_key = normalize_yandex_public_key(spec.public_key)
        response = requests.get(
            "https://cloud-api.yandex.com/v1/disk/public/resources/download",
            params={"public_key": public_key},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        href = payload.get("href")
        if not isinstance(href, str) or not href:
            raise RuntimeError(f"Yandex public link did not return href for {spec.artifact_id}.")
        return href
    if spec.source == "gdrive_file":
        if not spec.file_id:
            raise ValueError(f"Artifact {spec.artifact_id} is missing file_id.")
        return (
            "https://drive.usercontent.google.com/download"
            f"?id={quote_plus(spec.file_id)}&export=download&confirm=t"
        )
    raise ValueError(f"Unsupported artifact source for {spec.artifact_id}: {spec.source}")


def normalize_yandex_public_key(public_key: str) -> str:
    return public_key.replace("disk.yandex.ru", "disk.yandex.com")


def materialize(spec: ArtifactSpec) -> None:
    if not spec.archive_type:
        dest_path = spec.resolved_dest_path
        if dest_path is None:
            raise ValueError(f"Non-archive artifact {spec.artifact_id} must define dest_path.")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec.cache_path, dest_path)
        return

    if spec.archive_type not in {"tar", "tar.gz", "tar.zst"}:
        raise ValueError(f"Unsupported archive type for {spec.artifact_id}: {spec.archive_type}")
    with tempfile.TemporaryDirectory(prefix=f"{spec.artifact_id}-") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        extract_from_tar(spec, tmp_dir)
        if spec.resolved_dest_path is not None:
            source = tmp_dir / spec.archive_member
            if not source.is_file():
                raise FileNotFoundError(f"Archive member not found after extraction: {source}")
            dest_path = spec.resolved_dest_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest_path)
            return

        if spec.resolved_dest_dir is None:
            raise ValueError(f"Archive artifact {spec.artifact_id} must define a destination.")
        source_dir = tmp_dir / spec.archive_prefix
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Archive prefix not found after extraction: {source_dir}")
        dest_dir = spec.resolved_dest_dir
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, dest_dir)


def extract_from_tar(spec: ArtifactSpec, target_dir: Path) -> None:
    members: list[str] = []
    if spec.archive_member:
        members.append(spec.archive_member)
    if spec.archive_prefix:
        members.append(spec.archive_prefix)
    if not members:
        raise ValueError(f"Archive artifact {spec.artifact_id} must define member or prefix.")
    command = ["tar"]
    if spec.archive_type == "tar.zst":
        command.append("--zstd")
    elif spec.archive_type == "tar.gz":
        command.append("-z")
    command.extend(["-xf", str(spec.cache_path), "-C", str(target_dir), *members])
    subprocess.run(command, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
