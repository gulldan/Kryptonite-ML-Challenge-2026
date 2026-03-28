from pathlib import Path


def test_research_baseline_docs_exist_and_cover_epic_zero_outputs() -> None:
    epic_baseline = Path("docs/epic-00-research-baseline.md")
    rules_matrix = Path("docs/dataton-rules-matrix.md")
    task_contract = Path("docs/model-task-contract.md")
    verification_protocol = Path("docs/internal-verification-protocol.md")
    system_architecture = Path("docs/system-architecture-v1.md")
    experiment_matrix = Path("docs/experiment-matrix-v1.md")
    verification_protocol_config = Path("configs/eval/verification-protocol.toml")

    assert epic_baseline.is_file()
    assert rules_matrix.is_file()
    assert task_contract.is_file()
    assert verification_protocol.is_file()
    assert system_architecture.is_file()
    assert experiment_matrix.is_file()
    assert verification_protocol_config.is_file()

    epic_text = epic_baseline.read_text(encoding="utf-8")
    protocol_text = verification_protocol.read_text(encoding="utf-8")
    protocol_config_text = verification_protocol_config.read_text(encoding="utf-8")

    assert "KVA-467" in epic_text
    assert "KVA-479" in epic_text
    assert "KVA-480" in epic_text
    assert "KVA-481" in epic_text
    assert "KVA-482" in epic_text
    assert "KVA-483" in epic_text
    assert "## Validation" in epic_text
    assert (
        "build_verification_protocol.py --config "
        "configs/eval/verification-protocol.toml --require-complete"
        in epic_text
    )

    assert "## Builder" in protocol_text
    assert "## Validation Modes" in protocol_text
    assert "--require-complete" in protocol_text
    assert "prepare_ffsvc2022_surrogate.py" in protocol_text

    assert "--require-complete" in protocol_config_text


def test_research_baseline_docs_are_linked_from_repository_indexes() -> None:
    root_readme = Path("README.md").read_text(encoding="utf-8")
    docs_readme = Path("docs/README.md").read_text(encoding="utf-8")

    assert "docs/epic-00-research-baseline.md" in root_readme
    assert "docs/internal-verification-protocol.md" in root_readme
    assert "docs/epic-00-research-baseline.md" in docs_readme
    assert "docs/internal-verification-protocol.md" in docs_readme
