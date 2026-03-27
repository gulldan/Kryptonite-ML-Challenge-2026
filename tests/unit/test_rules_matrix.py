from __future__ import annotations

from pathlib import Path

import pytest

from kryptonite.data.rules_matrix import (
    build_rules_matrix_report,
    load_rules_matrix_plan,
    render_rules_matrix_markdown,
    write_rules_matrix_report,
)


def test_load_rules_matrix_plan_rejects_unknown_source_reference(tmp_path: Path) -> None:
    plan_path = tmp_path / "rules.toml"
    plan_path.write_text(
        """
title = "Rules Matrix"
reviewed_on = "2026-03-28"

[[sources]]
id = "official"
title = "Official site"
kind = "official"
url = "https://example.com"
reviewed_on = "2026-03-28"

[[items]]
id = "item-001"
name = "Questionable item"
category = "policy"
decision = "unknown"
confidence = "low"
repo_position = "Hold"
reasoning = "Need an answer."
owner = "Organizer"
clarification_channel = "Telegram"
next_checkpoint = "2026-04-11"
source_ids = ["missing-source"]

[[risks]]
id = "risk-001"
title = "Open question"
severity = "high"
description = "Still open."
owner = "Organizer"
clarification_channel = "Telegram"
mitigation = "Wait."
related_item_ids = ["item-001"]
"""
    )

    with pytest.raises(ValueError, match="unknown source ids"):
        load_rules_matrix_plan(plan_path)


def test_build_rules_matrix_report_resolves_repo_references(tmp_path: Path) -> None:
    doc_path = tmp_path / "docs" / "note.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("# note\n")

    plan_path = tmp_path / "rules.toml"
    plan_path.write_text(
        """
title = "Rules Matrix"
reviewed_on = "2026-03-28"
summary = ["One summary line."]

[[sources]]
id = "official"
title = "Official site"
kind = "official"
url = "https://example.com"
reviewed_on = "2026-03-28"
evidence = ["Criteria publish later."]

[[items]]
id = "item-001"
name = "Official data"
category = "official-data"
decision = "allow"
confidence = "high"
repo_position = "Use it."
reasoning = "Primary dataset."
owner = "Organizer"
clarification_channel = "Telegram"
next_checkpoint = "2026-04-11"
source_ids = ["official"]
repo_references = ["docs/note.md", "docs/missing.md"]

[[risks]]
id = "risk-001"
title = "Arrival risk"
severity = "medium"
description = "Could land late."
owner = "Organizer"
clarification_channel = "Telegram"
mitigation = "Keep adapters flexible."
related_item_ids = ["item-001"]
"""
    )

    plan = load_rules_matrix_plan(plan_path)
    report = build_rules_matrix_report(project_root=tmp_path, plan=plan, plan_path=plan_path)

    assert report.item_count == 1
    assert report.decision_counts == {"allow": 1, "deny": 0, "unknown": 0}
    assert report.open_question_count == 0
    assert report.risk_severity_counts == {"high": 0, "medium": 1, "low": 0}
    checks = report.entries[0].repo_reference_checks
    assert [check.exists for check in checks] == [True, False]
    assert checks[0].path_type == "file"
    assert checks[1].path_type == "missing"


def test_write_rules_matrix_report_writes_markdown_and_json(tmp_path: Path) -> None:
    plan_path = tmp_path / "rules.toml"
    plan_path.write_text(
        """
title = "Rules Matrix"
reviewed_on = "2026-03-28"
summary = ["Summary."]

[[sources]]
id = "official"
title = "Official site"
kind = "official"
url = "https://example.com"
reviewed_on = "2026-03-28"

[[items]]
id = "item-001"
name = "Official data"
category = "official-data"
decision = "allow"
confidence = "high"
repo_position = "Use it."
reasoning = "Primary dataset."
owner = "Organizer"
clarification_channel = "Telegram"
next_checkpoint = "2026-04-11"
source_ids = ["official"]

[[risks]]
id = "risk-001"
title = "Arrival risk"
severity = "low"
description = "Could land late."
owner = "Organizer"
clarification_channel = "Telegram"
mitigation = "Keep adapters flexible."
related_item_ids = ["item-001"]
"""
    )

    plan = load_rules_matrix_plan(plan_path)
    report = build_rules_matrix_report(project_root=tmp_path, plan=plan, plan_path=plan_path)
    written = write_rules_matrix_report(report=report, output_root=tmp_path / "out")

    markdown = Path(written.markdown_path).read_text()
    assert "## Rules Matrix" in markdown
    assert "Official data" in markdown
    assert "Arrival risk" in markdown
    assert render_rules_matrix_markdown(report) == markdown
    assert Path(written.json_path).exists()
