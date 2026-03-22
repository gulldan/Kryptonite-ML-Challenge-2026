from __future__ import annotations

import tarfile
from pathlib import Path

from kryptonite.data.acquisition import (
    _compute_checksum,
    _extract_archive,
    acquire_plan,
    load_acquisition_plan,
)


def test_load_acquisition_plan_parses_ffsvc_style_config(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(
        """
name = "ffsvc2022-surrogate"
dataset_root = "datasets/ffsvc2022-surrogate"
notes = ["gpu-only"]

[[artifacts]]
name = "dev_meta"
url = "https://example.com/dev_meta_list"
target_path = "metadata/dev_meta_list.txt"
checksum = "abc"
checksum_algorithm = "md5"
"""
    )

    plan = load_acquisition_plan(plan_path)

    assert plan.name == "ffsvc2022-surrogate"
    assert plan.dataset_root == "datasets/ffsvc2022-surrogate"
    assert plan.notes == ["gpu-only"]
    assert len(plan.artifacts) == 1
    assert plan.artifacts[0].checksum_algorithm == "md5"


def test_extract_archive_unpacks_tar_gz(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    payload = source_root / "hello.txt"
    payload.write_text("hello\n")

    archive_path = tmp_path / "archive.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        archive.add(payload, arcname="hello.txt")

    extract_root = tmp_path / "extract"
    error = _extract_archive(archive_path=archive_path, extract_root=extract_root)

    assert error is None
    assert (extract_root / "hello.txt").read_text() == "hello\n"


def test_acquire_plan_inspect_only_validates_existing_files(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets" / "ffsvc2022-surrogate"
    metadata_root = dataset_root / "metadata"
    metadata_root.mkdir(parents=True)
    target_path = metadata_root / "dev_meta_list.txt"
    target_path.write_text("speaker_a utt_001.wav\n")
    checksum = _compute_checksum(target_path, "md5")

    plan = load_acquisition_plan(
        _write_plan(
            tmp_path,
            checksum=checksum,
        )
    )
    report = acquire_plan(project_root=tmp_path, plan=plan, execute=False)

    assert report.passed is True
    assert report.artifacts[0].target_path == str(target_path)
    assert report.artifacts[0].checksum == checksum


def _write_plan(tmp_path: Path, *, checksum: str) -> Path:
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(
        f"""
name = "ffsvc2022-surrogate"
dataset_root = "datasets/ffsvc2022-surrogate"

[[artifacts]]
name = "dev_meta"
url = "https://example.com/dev_meta_list"
target_path = "metadata/dev_meta_list.txt"
checksum = "{checksum}"
checksum_algorithm = "md5"
"""
    )
    return plan_path
