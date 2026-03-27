from __future__ import annotations

import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.system_architecture import (
    SYSTEM_ARCHITECTURE_FORMAT_VERSION,
    build_system_architecture_contract,
    render_system_architecture_markdown,
    write_system_architecture_contract,
)


def test_system_architecture_contract_writes_machine_readable_snapshot(tmp_path: Path) -> None:
    config = load_project_config(config_path=Path("configs/base.toml"))

    contract = build_system_architecture_contract(config)
    written = write_system_architecture_contract(contract, project_root=tmp_path)

    assert contract.format_version == SYSTEM_ARCHITECTURE_FORMAT_VERSION
    assert contract.ticket_id == "KVA-482"
    assert contract.export_and_serve.boundary_mode == "encoder_only"
    assert contract.stages[0].stage_id == "raw_audio_ingest"
    assert any(stage.stage_id == "score_normalization" for stage in contract.stages)
    assert any(
        interface.interface_id == "encoder_export_boundary"
        for interface in contract.interface_points
    )
    assert Path(written.report_json_path).is_file()
    assert Path(written.report_markdown_path).is_file()

    payload = json.loads(Path(written.report_json_path).read_text(encoding="utf-8"))
    assert payload["ticket_id"] == "KVA-482"
    assert payload["export_and_serve"]["boundary_mode"] == "encoder_only"
    assert payload["stages"][2]["output_contract"].startswith("embedding")
    assert payload["logging_points"][2]["point_id"] == "serve_json_logs_and_metrics"

    markdown = Path(written.report_markdown_path).read_text(encoding="utf-8")
    assert "## Pipeline Diagram" in markdown
    assert "## Module Boundaries" in markdown
    assert "## Interfaces" in markdown
    assert "## Logging Points" in markdown


def test_checked_in_system_architecture_doc_is_linked() -> None:
    doc_path = Path("docs/system-architecture-v1.md")
    assert doc_path.is_file()

    doc_text = doc_path.read_text(encoding="utf-8")
    root_readme = Path("README.md").read_text(encoding="utf-8")
    docs_readme = Path("docs/README.md").read_text(encoding="utf-8")

    assert "## Pipeline Diagram" in doc_text
    assert "## Module Boundaries" in doc_text
    assert "## Interfaces" in doc_text
    assert "## Logging Points" in doc_text
    assert "scripts/build_system_architecture.py" in doc_text

    assert "docs/system-architecture-v1.md" in root_readme
    assert "docs/system-architecture-v1.md" in docs_readme


def test_rendered_system_architecture_mentions_runtime_and_observability() -> None:
    config = load_project_config(config_path=Path("configs/base.toml"))

    contract = build_system_architecture_contract(config)
    markdown = render_system_architecture_markdown(contract)

    assert "feature_statistics" in markdown
    assert "encoder_input" in markdown
    assert "/metrics" in markdown
    assert "verification_threshold_calibration.json" in markdown
