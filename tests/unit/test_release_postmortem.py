from __future__ import annotations

import json
from pathlib import Path

from kryptonite.serve import (
    build_release_postmortem,
    load_release_postmortem_config,
    write_release_postmortem,
)


def test_release_postmortem_writes_machine_readable_report(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    src_root = tmp_path / "src" / "kryptonite" / "serve"
    tests_root = tmp_path / "tests" / "e2e"
    docs_root.mkdir(parents=True, exist_ok=True)
    src_root.mkdir(parents=True, exist_ok=True)
    tests_root.mkdir(parents=True, exist_ok=True)

    (docs_root / "model-card.md").write_text("# Model Card\n", encoding="utf-8")
    (docs_root / "release-runbook.md").write_text("# Runbook\n", encoding="utf-8")
    (src_root / "inference_backend.py").write_text("SUPPORTED = {'feature_statistics'}\n")
    (tests_root / "test_inference_regression_suite.py").write_text("def test_stub():\n    pass\n")

    config_path = tmp_path / "release-postmortem.toml"
    config_path.write_text(
        """
title = "Fixture Release Postmortem"
release_id = "fixture-release"
summary = "Fixture summary"
output_root = "artifacts/release-postmortems/fixture-release"
validation_commands = ["uv run pytest tests/unit/test_release_postmortem.py"]

[[evidence]]
label = "model_card"
kind = "doc"
path = "docs/model-card.md"
description = "Model scope"

[[evidence]]
label = "runbook"
kind = "doc"
path = "docs/release-runbook.md"
description = "Runbook scope"

[[evidence]]
label = "backend_code"
kind = "code"
path = "src/kryptonite/serve/inference_backend.py"
description = "Backend implementation"

[[evidence]]
label = "regression_suite"
kind = "test"
path = "tests/e2e/test_inference_regression_suite.py"
description = "Regression coverage"

[[finding]]
area = "deploy"
outcome = "worked"
title = "Release docs exist"
detail = "The release surface is documented."
evidence = ["model_card", "runbook"]
related_issues = ["KVA-100"]

[[finding]]
area = "export"
outcome = "missed"
title = "Export path is still pending"
detail = "The runtime backend is not exported yet."
evidence = ["backend_code", "regression_suite"]

[[backlog_item]]
title = "Implement export"
priority = "P0"
disposition = "next_iteration"
area = "export"
rationale = "Need a real exported backend."
related_issue = "KVA-101"
dependencies = []

[[backlog_item]]
title = "De-scope stretch"
priority = "P3"
disposition = "de_scoped"
area = "teacher"
rationale = "Not on the critical path."
dependencies = []
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_release_postmortem_config(config_path=config_path)
    report = build_release_postmortem(
        config,
        config_path=config_path,
        project_root=tmp_path,
    )
    written = write_release_postmortem(report)

    assert report.summary.worked_count == 1
    assert report.summary.missed_count == 1
    assert report.summary.next_iteration_count == 1
    assert report.summary.de_scoped_count == 1
    assert Path(written.report_json_path).is_file()
    assert Path(written.report_markdown_path).is_file()

    payload = json.loads(Path(written.report_json_path).read_text(encoding="utf-8"))
    assert payload["summary"]["worked_count"] == 1
    assert payload["summary"]["next_iteration_count"] == 1
    assert payload["backlog_items"][0]["related_issue"] == "KVA-101"

    markdown = Path(written.report_markdown_path).read_text(encoding="utf-8")
    assert "## What Worked" in markdown
    assert "## Backlog v2" in markdown
    assert "Implement export" in markdown


def test_checked_in_release_postmortem_config_builds_against_repository() -> None:
    config_path = Path("configs/release/release-postmortem-v2.toml")

    config = load_release_postmortem_config(config_path=config_path)
    report = build_release_postmortem(config, config_path=config_path)

    assert report.release_id == "kryptonite-2026-release-v2"
    assert report.summary.next_iteration_count >= 3
    assert report.summary.de_scoped_count >= 3
    assert any(item.related_issue == "KVA-538" for item in report.backlog_items)
    assert any(item.title.startswith("The learned export path") for item in report.findings)
