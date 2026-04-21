"""Build VoxBlink v1 audio-only data from official annotations.

The official VoxBlink v1 release provides annotation resources, YouTube ids, and
timestamps, not audio files. This helper downloads the official resource bundle,
extracts timestamps, optionally clones the official scripts for provenance, and
builds 16 kHz mono WAV utterances without writing cropped video files.

VoxBlink2 is intentionally not supported by this script.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shutil
import signal
import subprocess
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DRIVE_URL = "https://drive.google.com/drive/folders/1vP8hyT_Zefj2d40JzHLAUJWy_dBjpA22"
OFFICIAL_SCRIPTS_REPO = "https://github.com/VoxBlink/ScriptsForVoxBlink.git"
DEFAULT_ROOT = Path("datasets/voxblink_v1")


@dataclass(frozen=True, slots=True)
class BuildConfig:
    root: Path
    mode: str
    workers: int
    max_speakers: int | None
    max_videos_per_speaker: int | None
    video_download_timeout_seconds: int
    output_dir: Path
    cache_dir: Path
    manifest_dir: Path
    base_train_manifest: Path | None
    force_resource_download: bool


@dataclass(frozen=True, slots=True)
class SpeakerJob:
    speaker_id: str
    video_ids: tuple[str, ...]


def main() -> None:
    args = _parse_args()
    root = Path(args.root)
    config = BuildConfig(
        root=root,
        mode=args.mode,
        workers=args.workers,
        max_speakers=args.max_speakers,
        max_videos_per_speaker=args.max_videos_per_speaker,
        video_download_timeout_seconds=args.video_download_timeout_seconds,
        output_dir=Path(args.output_dir) if args.output_dir else root / f"audio_{args.mode}",
        cache_dir=Path(args.cache_dir)
        if args.cache_dir
        else root / f"youtube_audio_cache_{args.mode}",
        manifest_dir=Path(args.manifest_dir),
        base_train_manifest=Path(args.base_train_manifest) if args.base_train_manifest else None,
        force_resource_download=args.force_resource_download,
    )

    if args.step in {"resources", "all"}:
        _prepare_resources(config)
    if args.step in {"audio", "all"}:
        _build_audio(config)
    if args.step in {"manifests", "all"}:
        _build_manifests(config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--mode", choices=("test", "clean", "full"), default="clean")
    parser.add_argument("--step", choices=("resources", "audio", "manifests", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--max-videos-per-speaker", type=int)
    parser.add_argument("--video-download-timeout-seconds", type=int, default=180)
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--manifest-dir", default="artifacts/manifests/voxblink_v1")
    parser.add_argument(
        "--base-train-manifest",
        default="",
        help="Optional participant train manifest to prepend into a mixed manifest.",
    )
    parser.add_argument("--force-resource-download", action="store_true")
    return parser.parse_args()


def _prepare_resources(config: BuildConfig) -> None:
    config.root.mkdir(parents=True, exist_ok=True)
    resource_dir = config.root / "resource"
    probe_dir = config.root / "resource_download_probe"
    if not resource_dir.exists() and probe_dir.exists():
        probe_dir.rename(resource_dir)

    if config.force_resource_download and resource_dir.exists():
        shutil.rmtree(resource_dir)
    if not _resource_bundle_present(resource_dir):
        resource_dir.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "uvx",
                "gdown",
                "--folder",
                DRIVE_URL,
                "-O",
                str(resource_dir),
                "--continue",
            ]
        )

    _extract_if_needed(resource_dir / "timestamp.tar.gz", resource_dir / "timestamp")
    _extract_if_needed(resource_dir / "video_tags.tar.gz", resource_dir / "video_tags_vb")
    if not (resource_dir / "video_tags").exists() and (resource_dir / "video_tags_vb").exists():
        (resource_dir / "video_tags").symlink_to("video_tags_vb")

    scripts_dir = config.root / "ScriptsForVoxBlink"
    if not scripts_dir.exists():
        _run(["git", "clone", "--depth", "1", OFFICIAL_SCRIPTS_REPO, str(scripts_dir)])
    link = scripts_dir / "resource"
    if not link.exists():
        link.symlink_to("../resource")

    inventory = {
        "drive_url": DRIVE_URL,
        "official_scripts_repo": OFFICIAL_SCRIPTS_REPO,
        "resource_dir": str(resource_dir),
        "scripts_dir": str(scripts_dir),
        "files": _resource_inventory_files(resource_dir),
    }
    (config.root / "resource_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(inventory, indent=2, sort_keys=True), flush=True)


def _resource_bundle_present(resource_dir: Path) -> bool:
    required = [
        "data/utt_clean.txt",
        "data/utt_full.txt",
        "meta/utt2dur",
        "video_list/spk2videos_clean",
        "video_list/spk2videos_full",
        "video_list/spk2videos_test",
        "timestamp.tar.gz",
    ]
    return all((resource_dir / item).exists() for item in required)


def _resource_inventory_files(resource_dir: Path) -> list[str]:
    paths: list[str] = []
    for relative_dir in ("data", "meta", "video_list"):
        directory = resource_dir / relative_dir
        if directory.exists():
            paths.extend(
                str(path.relative_to(resource_dir))
                for path in sorted(directory.iterdir())
                if path.is_file() or path.is_symlink()
            )
    for filename in ("timestamp.tar.gz", "video_tags.tar.gz"):
        if (resource_dir / filename).exists():
            paths.append(filename)
    for dirname in ("timestamp", "video_tags"):
        if (resource_dir / dirname).exists():
            paths.append(f"{dirname}/")
    return sorted(paths)


def _extract_if_needed(archive_path: Path, expected_dir: Path) -> None:
    if expected_dir.exists():
        return
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(archive_path.parent)


def _build_audio(config: BuildConfig) -> None:
    _prepare_resources(config)
    ffmpeg_path = _resolve_ffmpeg()
    jobs = _load_jobs(config)
    if not jobs:
        raise SystemExit("No VoxBlink jobs selected.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.root / f"audio_build_{config.mode}_summary.jsonl"
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, config.workers)) as pool:
        futures = [
            pool.submit(
                _process_speaker,
                job,
                str(config.root),
                str(config.output_dir),
                str(config.cache_dir),
                ffmpeg_path,
                config.video_download_timeout_seconds,
            )
            for job in jobs
        ]
        with summary_path.open("a", encoding="utf-8") as handle:
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                print(json.dumps(row, sort_keys=True), flush=True)


def _load_jobs(config: BuildConfig) -> list[SpeakerJob]:
    list_path = config.root / "resource" / "video_list" / f"spk2videos_{config.mode}"
    if not list_path.exists():
        raise FileNotFoundError(list_path)
    jobs: list[SpeakerJob] = []
    with list_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            video_ids = tuple(parts[1:])
            if config.max_videos_per_speaker is not None:
                video_ids = video_ids[: config.max_videos_per_speaker]
            jobs.append(SpeakerJob(speaker_id=parts[0], video_ids=video_ids))
            if config.max_speakers is not None and len(jobs) >= config.max_speakers:
                break
    return jobs


def _process_speaker(
    job: SpeakerJob,
    root: str,
    output_dir: str,
    cache_dir: str,
    ffmpeg_path: str,
    video_download_timeout_seconds: int,
) -> dict[str, Any]:
    root_path = Path(root)
    output_path = Path(output_dir)
    cache_path = Path(cache_dir)
    speaker_stats = Counter[str]()
    for video_id in job.video_ids:
        timestamp_video_dir = root_path / "resource" / "timestamp" / job.speaker_id / video_id
        if not timestamp_video_dir.exists():
            speaker_stats["missing_timestamp_video"] += 1
            continue
        audio_source = _download_audio(
            job.speaker_id,
            video_id,
            cache_path,
            timeout_seconds=video_download_timeout_seconds,
        )
        if not audio_source:
            speaker_stats["download_failed"] += 1
            continue
        segment_dir = output_path / job.speaker_id / video_id
        segment_dir.mkdir(parents=True, exist_ok=True)
        for timestamp_file in sorted(timestamp_video_dir.glob("*.txt")):
            output_wav = segment_dir / f"{timestamp_file.stem}.wav"
            if output_wav.exists() and output_wav.stat().st_size > 0:
                speaker_stats["segments_existing"] += 1
                continue
            start_frame, end_frame = _timestamp_frame_range(timestamp_file)
            if start_frame is None or end_frame is None or end_frame <= start_frame:
                speaker_stats["bad_timestamp"] += 1
                continue
            start_seconds = start_frame / 25.0
            end_seconds = end_frame / 25.0
            cmd = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start_seconds:.3f}",
                "-to",
                f"{end_seconds:.3f}",
                "-i",
                str(audio_source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                str(output_wav),
                "-y",
            ]
            result = subprocess.run(cmd, check=False)
            if result.returncode == 0 and output_wav.exists() and output_wav.stat().st_size > 0:
                speaker_stats["segments_written"] += 1
            else:
                speaker_stats["ffmpeg_failed"] += 1
                output_wav.unlink(missing_ok=True)
    return {"speaker_id": job.speaker_id, **dict(speaker_stats)}


def _download_audio(
    speaker_id: str,
    video_id: str,
    cache_dir: Path,
    *,
    timeout_seconds: int,
) -> Path | None:
    speaker_cache = cache_dir / speaker_id
    speaker_cache.mkdir(parents=True, exist_ok=True)
    marker = speaker_cache / f"{video_id}.path"
    if marker.exists():
        cached = Path(marker.read_text(encoding="utf-8").strip())
        if cached.exists() and cached.stat().st_size > 0:
            return cached
    output_template = speaker_cache / f"{video_id}.%(ext)s"
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "uvx",
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "--socket-timeout",
        "30",
        "--retries",
        "2",
        "--fragment-retries",
        "2",
        "--no-playlist",
        "--ignore-errors",
        "--no-warnings",
        "--quiet",
        "-o",
        str(output_template),
        url,
    ]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        return None
    candidates = sorted(
        path
        for path in speaker_cache.glob(f"{video_id}.*")
        if path.suffix not in {".part", ".path", ".tmp", ".ytdl"}
        and ".part" not in path.name
        and path.stat().st_size > 0
    )
    if returncode != 0:
        return None
    if not candidates:
        return None
    selected = candidates[0]
    marker.write_text(str(selected), encoding="utf-8")
    return selected


def _timestamp_frame_range(path: Path) -> tuple[int | None, int | None]:
    frames: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < 5:
                continue
            parts = line.strip().split("\t")
            if not parts or not parts[0].strip():
                continue
            try:
                frames.append(int(parts[0]))
            except ValueError:
                continue
    if not frames:
        return None, None
    return min(frames), max(frames)


def _resolve_ffmpeg() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    result = subprocess.run(
        [
            "uv",
            "run",
            "--with",
            "imageio-ffmpeg",
            "python",
            "-c",
            "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip().splitlines()[-1]


def _build_manifests(config: BuildConfig) -> None:
    audio_root = config.output_dir
    rows = []
    for audio_path in sorted(audio_root.rglob("*.wav")):
        rel = audio_path.relative_to(audio_root)
        if len(rel.parts) < 3:
            continue
        speaker_id, video_id, filename = rel.parts[0], rel.parts[1], rel.parts[2]
        utterance_id = f"{speaker_id}-{video_id}-{Path(filename).stem}"
        rows.append(
            {
                "schema_version": "kryptonite.manifest.v1",
                "record_type": "utterance",
                "dataset": f"voxblink_v1_{config.mode}",
                "source_dataset": "voxblink_v1",
                "speaker_id": f"voxblink_{speaker_id}",
                "utterance_id": f"voxblink_{utterance_id}",
                "split": "external_train",
                "audio_path": audio_path.as_posix(),
                "channel": "mono",
            }
        )

    config.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.manifest_dir / f"voxblink_v1_{config.mode}_manifest.jsonl"
    csv_path = config.manifest_dir / f"voxblink_v1_{config.mode}_manifest.csv"
    mixed_path = config.manifest_dir / f"voxblink_v1_{config.mode}_mixed_train_manifest.jsonl"
    summary_path = config.manifest_dir / f"voxblink_v1_{config.mode}_summary.json"
    _write_jsonl(manifest_path, rows)
    _write_csv(csv_path, rows)

    mixed_row_count = 0
    if config.base_train_manifest:
        mixed_row_count = _write_mixed_manifest(
            mixed_path,
            base_train_manifest=config.base_train_manifest,
            external_manifest=manifest_path,
        )
    summary = {
        "mode": config.mode,
        "audio_root": str(audio_root),
        "manifest_path": str(manifest_path),
        "mixed_train_manifest": str(mixed_path) if config.base_train_manifest else "",
        "row_count": len(rows),
        "speaker_count": len({row["speaker_id"] for row in rows}),
        "mixed_row_count": mixed_row_count,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_mixed_manifest(
    path: Path,
    *,
    base_train_manifest: Path,
    external_manifest: Path,
) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for source in (base_train_manifest, external_manifest):
            for line in source.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    output.write(line.rstrip() + "\n")
                    count += 1
    return count


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _kill_process_group(process: subprocess.Popen[Any]) -> None:
    """Clean up timed out uvx/yt-dlp process trees."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait()


if __name__ == "__main__":
    main()
