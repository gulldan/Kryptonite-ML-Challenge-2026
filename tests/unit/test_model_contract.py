from __future__ import annotations

import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.model_contract import (
    MODEL_TASK_CONTRACT_FORMAT_VERSION,
    build_model_task_contract,
    render_model_task_contract_markdown,
    write_model_task_contract,
)


def test_model_task_contract_writes_machine_readable_snapshot(tmp_path: Path) -> None:
    config = load_project_config(config_path=Path("configs/base.toml"))

    contract = build_model_task_contract(config)
    written = write_model_task_contract(contract, project_root=tmp_path)

    assert contract.format_version == MODEL_TASK_CONTRACT_FORMAT_VERSION
    assert contract.primary_task_mode == "verification"
    assert contract.raw_audio_contract.sample_rate_hz == 16000
    assert contract.embedding_contract.boundary_mode == "encoder_only"
    assert any(mode.mode_id == "closed_set_identification" for mode in contract.task_modes)
    assert any(mode.mode_id == "open_set_identification" for mode in contract.task_modes)
    assert Path(written.report_json_path).is_file()
    assert Path(written.report_markdown_path).is_file()

    payload = json.loads(Path(written.report_json_path).read_text(encoding="utf-8"))
    assert payload["ticket_id"] == "KVA-480"
    assert payload["embedding_contract"]["input_signature"].startswith("encoder_input")
    assert payload["embedding_contract"]["output_signature"].startswith("embedding")
    assert payload["task_modes"][0]["mode_id"] == "verification"

    markdown = Path(written.report_markdown_path).read_text(encoding="utf-8")
    assert "## Decision" in markdown
    assert "## Trial Types" in markdown
    assert "closed_set_identification" in markdown
    assert "open_set_identification" in markdown


def test_checked_in_model_task_contract_doc_is_linked() -> None:
    doc_path = Path("docs/model-task-contract.md")
    assert doc_path.is_file()

    doc_text = doc_path.read_text(encoding="utf-8")
    root_readme = Path("README.md").read_text(encoding="utf-8")
    docs_readme = Path("docs/README.md").read_text(encoding="utf-8")

    assert "## Decision" in doc_text
    assert "closed-set identification" in doc_text
    assert "open-set identification" in doc_text
    assert "## Expected Artifacts" in doc_text
    assert "scripts/build_model_task_contract.py" in doc_text

    assert "docs/model-task-contract.md" in root_readme
    assert "docs/model-task-contract.md" in docs_readme


def test_rendered_model_task_contract_markdown_mentions_current_runtime() -> None:
    config = load_project_config(config_path=Path("configs/base.toml"))

    contract = build_model_task_contract(config)
    markdown = render_model_task_contract_markdown(contract)

    assert "feature_statistics" in markdown
    assert "verification_pair" in markdown
    assert "verification_threshold_calibration.json" in markdown
