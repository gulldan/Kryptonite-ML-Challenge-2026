"""Download external datasets for training and evaluation.

Usage:
    uv run python research/scripts/download_datasets.py --list
    uv run python research/scripts/download_datasets.py --dataset musan
    uv run python research/scripts/download_datasets.py --dataset cn-celeb
    uv run python research/scripts/download_datasets.py --dataset rirs-noises
    uv run python research/scripts/download_datasets.py --dataset all
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)

DATASETS_ROOT = Path("datasets")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    url: str
    mirrors: tuple[str, ...]
    archive_name: str
    extract_dir: str
    size_human: str
    md5: str | None
    license: str

    def local_path(self) -> Path:
        return DATASETS_ROOT / self.extract_dir

    def is_downloaded(self) -> bool:
        return self.local_path().exists()


DATASETS: dict[str, DatasetSpec] = {
    "musan": DatasetSpec(
        name="MUSAN",
        description="Noise corpus (speech/music/noise) for augmentation. OpenSLR 17.",
        url="https://huggingface.co/datasets/thusinh1969/musan/resolve/main/musan/musan.zip?download=true",
        mirrors=(),
        archive_name="musan.zip",
        extract_dir="musan",
        size_human="~11 GB",
        md5=None,
        license="CC BY 4.0 / various",
    ),
    "cn-celeb": DatasetSpec(
        name="CN-Celeb (v2)",
        description="Chinese celebrity speaker recognition. ~1000 speakers. OpenSLR 82.",
        url="https://www.openslr.org/resources/82/cn-celeb_v2.tar.gz",
        mirrors=(
            "https://openslr.trmal.net/resources/82/cn-celeb_v2.tar.gz",
            "https://openslr.elda.org/resources/82/cn-celeb_v2.tar.gz",
            "https://openslr.magicdatatech.com/resources/82/cn-celeb_v2.tar.gz",
        ),
        archive_name="cn-celeb_v2.tar.gz",
        extract_dir="CN-Celeb_flac",
        size_human="~22 GB",
        md5="7ab1b214028a7439e26608b2d5a0336c",
        license="CC BY-SA 4.0",
    ),
    "rirs-noises": DatasetSpec(
        name="RIRs and Noises",
        description="Room impulse responses and isotropic noises. OpenSLR 28.",
        url="https://huggingface.co/datasets/EaseZh/rirs_noises/resolve/main/rirs_noises.zip?download=true",
        mirrors=(
            "https://www.openslr.org/resources/28/rirs_noises.zip",
            "https://openslr.trmal.net/resources/28/rirs_noises.zip",
            "https://openslr.elda.org/resources/28/rirs_noises.zip",
        ),
        archive_name="rirs_noises.zip",
        extract_dir="RIRS_NOISES",
        size_human="~3.6 GB",
        md5=None,
        license="Apache 2.0",
    ),
}


def _download_file(spec: DatasetSpec) -> Path:
    """Download archive using wget with mirror fallback."""
    archive_path = DATASETS_ROOT / spec.archive_name
    if archive_path.exists():
        typer.echo(f"  Archive already exists: {archive_path}")
        return archive_path

    urls = [spec.url, *spec.mirrors]
    for i, url in enumerate(urls):
        label = "primary" if i == 0 else f"mirror {i}"
        typer.echo(f"  Trying {label}: {url}")
        result = subprocess.run(
            [
                "wget",
                "--no-check-certificate",
                "--retry-connrefused",
                "--waitretry=3",
                "--timeout=120",
                "-t",
                "3",
                "-c",
                url,
                "-O",
                archive_path.name,
            ],
            cwd=str(DATASETS_ROOT),
        )
        if result.returncode == 0 and archive_path.exists() and archive_path.stat().st_size > 0:
            return archive_path
        typer.echo(f"  Failed from {label}, trying next...")

    typer.echo("  ERROR: All download URLs failed.", err=True)
    raise typer.Exit(1)


def _verify_md5(path: Path, expected: str) -> bool:
    typer.echo("  Verifying MD5...")
    md5 = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            md5.update(chunk)
    actual = md5.hexdigest()
    if actual != expected:
        typer.echo(f"  MD5 mismatch: expected {expected}, got {actual}", err=True)
        return False
    typer.echo(f"  MD5 OK: {actual}")
    return True


def _extract_archive(archive_path: Path) -> None:
    name = archive_path.name
    typer.echo(f"  Extracting {name}...")
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(path=DATASETS_ROOT)
    elif name.endswith(".zip"):
        import zipfile

        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(path=DATASETS_ROOT)
    else:
        typer.echo(f"  Unknown archive format: {name}", err=True)
        raise typer.Exit(1)


def download_dataset(key: str, *, force: bool = False) -> None:
    spec = DATASETS[key]
    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"Dataset: {spec.name}")
    typer.echo(f"Description: {spec.description}")
    typer.echo(f"Size: {spec.size_human}")
    typer.echo(f"License: {spec.license}")
    typer.echo(f"{'=' * 60}")

    if spec.is_downloaded() and not force:
        typer.echo(f"  Already downloaded at {spec.local_path()}")
        typer.echo("  Use --force to re-download.")
        return

    if spec.is_downloaded() and force:
        typer.echo(f"  Removing existing {spec.local_path()}...")
        shutil.rmtree(spec.local_path())

    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    archive_path = _download_file(spec)

    if spec.md5:
        if not _verify_md5(archive_path, spec.md5):
            typer.echo("  Checksum failed. Archive may be corrupted.", err=True)
            raise typer.Exit(1)

    _extract_archive(archive_path)

    if spec.local_path().exists():
        typer.echo(f"  Removing archive {archive_path.name}...")
        archive_path.unlink(missing_ok=True)
        typer.echo(f"  Done: {spec.local_path()}")
    else:
        typer.echo(f"  WARNING: Expected directory {spec.local_path()} not found after extraction.")
        typer.echo(f"  Archive kept at {archive_path}")


@app.command()
def main(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        help="Dataset to download: musan, cn-celeb, rirs-noises, or 'all'.",
        case_sensitive=False,
    ),
    force: bool = typer.Option(False, "--force", help="Force re-download even if exists."),
    list_only: bool = typer.Option(False, "--list", help="List available datasets and exit."),
) -> None:
    if list_only:
        typer.echo("\nAvailable datasets:\n")
        for key, spec in DATASETS.items():
            status = "DOWNLOADED" if spec.is_downloaded() else "not downloaded"
            typer.echo(f"  {key:15s}  {spec.size_human:>8s}  [{status}]  {spec.description}")
        typer.echo(f"\nDatasets root: {DATASETS_ROOT.resolve()}")
        return

    normalized = dataset.strip().lower()
    if normalized == "all":
        for key in DATASETS:
            download_dataset(key, force=force)
    elif normalized in DATASETS:
        download_dataset(normalized, force=force)
    else:
        typer.echo(f"Unknown dataset: {dataset!r}. Available: {', '.join(DATASETS)}, all")
        raise typer.Exit(1)

    typer.echo("\nAll requested downloads complete.")


if __name__ == "__main__":
    app()
