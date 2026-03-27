from pathlib import Path


def test_release_docs_exist_and_cover_required_topics() -> None:
    model_card = Path("docs/model-card.md")
    runbook = Path("docs/release-runbook.md")
    submission_bundle = Path("docs/submission-release-bundle.md")
    postmortem = Path("docs/release-postmortem.md")
    postmortem_config = Path("configs/release/release-postmortem-v2.toml")

    assert model_card.is_file()
    assert runbook.is_file()
    assert submission_bundle.is_file()
    assert postmortem.is_file()
    assert postmortem_config.is_file()

    model_card_text = model_card.read_text(encoding="utf-8")
    runbook_text = runbook.read_text(encoding="utf-8")
    submission_bundle_text = submission_bundle.read_text(encoding="utf-8")
    postmortem_text = postmortem.read_text(encoding="utf-8")
    postmortem_config_text = postmortem_config.read_text(encoding="utf-8")

    assert "## Training Data Classes" in model_card_text
    assert "## Bias And Robustness Caveats" in model_card_text
    assert "## Limitations And Risks" in model_card_text
    assert "## Deployment Notes" in model_card_text

    assert "## Preflight Checklist" in runbook_text
    assert "## Monitoring And Incident Triage" in runbook_text
    assert "## Rollback Procedure" in runbook_text
    assert "verification_threshold_calibration.json" in runbook_text

    assert "## Bundle Modes" in submission_bundle_text
    assert "scripts/build_submission_bundle.py" in submission_bundle_text
    assert "release_freeze.json" in submission_bundle_text
    assert "data_manifest_paths" in submission_bundle_text
    assert "## Validation" in submission_bundle_text

    assert "## What Worked" in postmortem_text
    assert "## What Did Not Ship" in postmortem_text
    assert "## Backlog v2" in postmortem_text
    assert "feature_statistics" in postmortem_text
    assert "KVA-538" in postmortem_text
    assert "scripts/build_release_postmortem.py" in postmortem_text

    assert "KVA-544" in postmortem_config_text
    assert "next_iteration" in postmortem_config_text
    assert "de_scoped" in postmortem_config_text


def test_release_docs_are_linked_from_repository_indexes() -> None:
    root_readme = Path("README.md").read_text(encoding="utf-8")
    docs_readme = Path("docs/README.md").read_text(encoding="utf-8")

    assert "docs/model-card.md" in root_readme
    assert "docs/release-runbook.md" in root_readme
    assert "docs/submission-release-bundle.md" in root_readme
    assert "docs/release-postmortem.md" in root_readme
    assert "docs/model-card.md" in docs_readme
    assert "docs/release-runbook.md" in docs_readme
    assert "docs/submission-release-bundle.md" in docs_readme
    assert "docs/release-postmortem.md" in docs_readme
