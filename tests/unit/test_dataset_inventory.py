from __future__ import annotations

from pathlib import Path

from kryptonite.data.inventory import (
    build_dataset_inventory_report,
    load_dataset_inventory_plan,
    render_dataset_inventory_markdown,
    write_dataset_inventory_report,
)


def test_load_dataset_inventory_plan_parses_sources(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)

    plan = load_dataset_inventory_plan(plan_path)

    assert plan.notes == ["policy note"]
    assert len(plan.sources) == 2
    assert plan.sources[0].id == "musan"
    assert plan.sources[1].status == "blocked"
    assert plan.sources[0].restrictions == ["keep provenance"]


def test_build_dataset_inventory_report_tracks_local_state(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    (tmp_path / "datasets" / "musan").mkdir(parents=True)

    plan = load_dataset_inventory_plan(plan_path)
    report = build_dataset_inventory_report(
        project_root=tmp_path,
        plan=plan,
        plan_path=plan_path,
    )

    assert report.source_count == 2
    assert report.status_counts == {"approved": 1, "conditional": 0, "blocked": 1}
    assert report.local_state_counts == {"partial": 1, "missing": 1}
    assert report.entries[0].source.id == "musan"
    assert report.entries[0].local_state == "partial"


def test_write_dataset_inventory_report_emits_json_and_markdown(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    plan = load_dataset_inventory_plan(plan_path)
    report = build_dataset_inventory_report(
        project_root=tmp_path,
        plan=plan,
        plan_path=plan_path,
    )

    written = write_dataset_inventory_report(
        report=report,
        output_root=tmp_path / "artifacts" / "reports" / "dataset-inventory",
    )

    json_path = Path(written.json_path)
    markdown_path = Path(written.markdown_path)
    assert json_path.exists()
    assert markdown_path.exists()

    markdown = render_dataset_inventory_markdown(report)
    assert "## Source Matrix" in markdown
    assert "MUSAN" in markdown
    assert "Dataton raw data" in markdown


def _write_plan(tmp_path: Path) -> Path:
    plan_path = tmp_path / "inventory.toml"
    plan_path.write_text(
        """
notes = ["policy note"]

[[sources]]
id = "musan"
name = "MUSAN"
kind = "noise-corpus"
status = "approved"
scopes = ["noise", "augmentation"]
license = "CC BY 4.0"
access = "public"
domain = "noise"
leakage_risk = "low"
rationale = "usable"
source_urls = ["https://example.com/musan"]
expected_paths = ["datasets/musan", "datasets/musan/metadata.json"]
restrictions = ["keep provenance"]

[[sources]]
id = "dataton"
name = "Dataton raw data"
kind = "challenge-dataset"
status = "blocked"
scopes = ["train"]
license = "pending"
access = "missing"
domain = "target"
leakage_risk = "unknown"
rationale = "not released"
expected_paths = ["datasets/dataton"]
"""
    )
    return plan_path
