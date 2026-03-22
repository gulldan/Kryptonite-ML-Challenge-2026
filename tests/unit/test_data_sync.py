from __future__ import annotations

from pathlib import Path

from kryptonite.data.sync import collect_local_inventory, load_sync_plan, resolve_remote_path


def test_collect_local_inventory_builds_catalog_checksum_for_directory(tmp_path: Path) -> None:
    dataset_root = tmp_path / "datasets"
    speaker_alpha = dataset_root / "speaker_alpha"
    speaker_bravo = dataset_root / "speaker_bravo"
    speaker_alpha.mkdir(parents=True)
    speaker_bravo.mkdir(parents=True)
    (speaker_alpha / "enroll.wav").write_bytes(b"alpha")
    (speaker_bravo / "test.wav").write_bytes(b"bravo")

    snapshot = collect_local_inventory(
        path=dataset_root,
        path_type="dir",
        checksum_mode="catalog",
        sample_limit=10,
    )

    assert snapshot.exists is True
    assert snapshot.error is None
    assert snapshot.file_count == 2
    assert snapshot.total_bytes == 10
    assert snapshot.samples == ["speaker_alpha/enroll.wav", "speaker_bravo/test.wav"]
    assert snapshot.checksum is not None
    assert len(snapshot.checksum) == 64


def test_collect_local_inventory_sha256_changes_with_file_content(tmp_path: Path) -> None:
    manifests_root = tmp_path / "manifests"
    manifests_root.mkdir()
    manifest_path = manifests_root / "demo_manifest.jsonl"
    manifest_path.write_text('{"speaker_id":"speaker_alpha"}\n')

    first_snapshot = collect_local_inventory(
        path=manifests_root,
        path_type="dir",
        checksum_mode="sha256",
        sample_limit=10,
    )

    manifest_path.write_text('{"speaker_id":"speaker_bravo"}\n')
    second_snapshot = collect_local_inventory(
        path=manifests_root,
        path_type="dir",
        checksum_mode="sha256",
        sample_limit=10,
    )

    assert first_snapshot.checksum != second_snapshot.checksum


def test_load_sync_plan_parses_remote_and_payloads(tmp_path: Path) -> None:
    plan_path = tmp_path / "gpu-server.toml"
    plan_path.write_text(
        """
notes = ["demo only"]

[remote]
host = "gpu-server"
project_root = "/mnt/storage/Kryptonite-ML-Challenge-2026"
ssh_options = ["-o", "BatchMode=yes"]

[report]
local_path = "artifacts/reports/gpu-server-data-sync.json"
remote_path = "artifacts/reports/gpu-server-data-sync.json"

[[payloads]]
name = "datasets"
source = "datasets"
target = "datasets"
path_type = "dir"
checksum_mode = "catalog"
required = true
"""
    )

    plan = load_sync_plan(plan_path)

    assert plan.remote.host == "gpu-server"
    assert plan.report.remote_path == "artifacts/reports/gpu-server-data-sync.json"
    assert plan.notes == ["demo only"]
    assert len(plan.payloads) == 1
    assert plan.payloads[0].checksum_mode == "catalog"


def test_resolve_remote_path_keeps_absolute_targets() -> None:
    assert (
        resolve_remote_path("/mnt/storage/Kryptonite-ML-Challenge-2026", "datasets")
        == "/mnt/storage/Kryptonite-ML-Challenge-2026/datasets"
    )
    assert (
        resolve_remote_path("/mnt/storage/Kryptonite-ML-Challenge-2026", "/tmp/data") == "/tmp/data"
    )
