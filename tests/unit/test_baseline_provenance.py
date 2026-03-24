from __future__ import annotations

from pathlib import Path

import pytest

from kryptonite.training import load_campp_baseline_config
from kryptonite.training.baseline_config import BaselineProvenanceConfig
from kryptonite.training.speaker_baseline import (
    EmbeddingExportSummary,
    EpochSummary,
    ScoreSummary,
    TrainingSummary,
    render_markdown_report,
)


@pytest.mark.parametrize(
    ("teacher_resources", "pretrained_resources", "message"),
    [
        (("teacher://remote",), (), "teacher_resources"),
        ((), ("pretrained://remote",), "pretrained_resources"),
    ],
)
def test_restricted_rules_provenance_rejects_external_dependencies(
    teacher_resources: tuple[str, ...],
    pretrained_resources: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BaselineProvenanceConfig(
            ruleset="restricted-rules",
            initialization="from_scratch",
            teacher_resources=teacher_resources,
            pretrained_resources=pretrained_resources,
        )


def test_campp_loader_reads_restricted_rules_provenance(tmp_path: Path) -> None:
    config_path = tmp_path / "campp-restricted.toml"
    config_path.write_text(
        "\n".join(
            [
                f'base_config = "{Path("configs/base.toml").resolve().as_posix()}"',
                "project_overrides = [",
                f"  'paths.project_root=\"{tmp_path.as_posix()}\"',",
                "]",
                "",
                "[provenance]",
                'ruleset = "restricted-rules"',
                'initialization = "from_scratch"',
                "teacher_resources = []",
                "pretrained_resources = []",
                'notes = ["restricted fallback"]',
                "",
            ]
        )
    )

    config = load_campp_baseline_config(config_path=config_path, env_file=tmp_path / ".env")

    assert config.provenance.ruleset == "restricted-rules"
    assert config.provenance.initialization == "from_scratch"
    assert config.provenance.notes == ("restricted fallback",)


def test_render_markdown_report_includes_provenance_notes(tmp_path: Path) -> None:
    project_root = tmp_path
    output_root = project_root / "artifacts" / "baselines" / "campp" / "run-001"
    output_root.mkdir(parents=True, exist_ok=True)

    report = render_markdown_report(
        title="CAM++ Baseline Report",
        provenance=BaselineProvenanceConfig(
            ruleset="restricted-rules",
            initialization="from_scratch",
            notes=("repo-native fallback",),
        ),
        training_summary=TrainingSummary(
            device="cpu",
            train_manifest="artifacts/manifests/train.jsonl",
            dev_manifest="artifacts/manifests/dev.jsonl",
            provenance_ruleset="restricted-rules",
            provenance_initialization="from_scratch",
            speaker_count=2,
            train_row_count=8,
            dev_row_count=4,
            checkpoint_path=str(output_root / "campp_encoder.pt"),
            epochs=(EpochSummary(epoch=1, mean_loss=1.23, accuracy=0.5, learning_rate=0.1),),
        ),
        embedding_summary=EmbeddingExportSummary(
            manifest_path="artifacts/manifests/dev.jsonl",
            embedding_dim=32,
            utterance_count=4,
            speaker_count=2,
            embeddings_path=str(output_root / "dev_embeddings.npz"),
            metadata_jsonl_path=str(output_root / "dev_embedding_metadata.jsonl"),
            metadata_parquet_path=str(output_root / "dev_embedding_metadata.parquet"),
        ),
        score_summary=ScoreSummary(
            trials_path=str(output_root / "dev_trials.jsonl"),
            scores_path=str(output_root / "dev_scores.jsonl"),
            trial_count=4,
            positive_count=2,
            negative_count=2,
            missing_embedding_count=0,
            mean_positive_score=0.5,
            mean_negative_score=0.1,
            score_gap=0.4,
        ),
        output_root=output_root,
        project_root=project_root,
    )

    assert "- Ruleset: `restricted-rules`" in report
    assert "## Provenance" in report
    assert "- Note: repo-native fallback" in report
