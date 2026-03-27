from pathlib import Path


def test_release_docs_exist_and_cover_required_topics() -> None:
    model_card = Path("docs/model-card.md")
    runbook = Path("docs/release-runbook.md")
    submission_bundle = Path("docs/submission-release-bundle.md")

    assert model_card.is_file()
    assert runbook.is_file()
    assert submission_bundle.is_file()

    model_card_text = model_card.read_text(encoding="utf-8")
    runbook_text = runbook.read_text(encoding="utf-8")
    submission_bundle_text = submission_bundle.read_text(encoding="utf-8")

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


def test_release_docs_are_linked_from_repository_indexes() -> None:
    root_readme = Path("README.md").read_text(encoding="utf-8")
    docs_readme = Path("docs/README.md").read_text(encoding="utf-8")

    assert "docs/model-card.md" in root_readme
    assert "docs/release-runbook.md" in root_readme
    assert "docs/submission-release-bundle.md" in root_readme
    assert "docs/model-card.md" in docs_readme
    assert "docs/release-runbook.md" in docs_readme
    assert "docs/submission-release-bundle.md" in docs_readme
