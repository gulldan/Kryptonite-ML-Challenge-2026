from __future__ import annotations

from pathlib import Path

from kryptonite.training import load_teacher_peft_config


def test_teacher_peft_loader_reads_defaults_and_pretrained_provenance(tmp_path: Path) -> None:
    config_path = tmp_path / "teacher-peft.toml"
    config_path.write_text(
        "\n".join(
            [
                f'base_config = "{Path("configs/base.toml").resolve().as_posix()}"',
                "project_overrides = [",
                f"  'paths.project_root=\"{tmp_path.as_posix()}\"',",
                "]",
                "",
                "[data]",
                'train_manifest = "artifacts/manifests/train.jsonl"',
                'dev_manifest = "artifacts/manifests/dev.jsonl"',
                'output_root = "artifacts/baselines/teacher-peft-test"',
                'checkpoint_name = "teacher_peft"',
                "",
                "[model]",
                'model_id = "facebook/w2v-bert-2.0"',
                "embedding_dim = 192",
                "",
                "[adapter]",
                "rank = 8",
                'target_modules = ["all-linear"]',
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_teacher_peft_config(config_path=config_path, env_file=tmp_path / ".env")

    assert config.model.model_id == "facebook/w2v-bert-2.0"
    assert config.model.resolved_feature_extractor_id == "facebook/w2v-bert-2.0"
    assert config.adapter.target_modules == ("all-linear",)
    assert config.provenance.ruleset == "standard"
    assert config.provenance.initialization == "pretrained"
    assert config.provenance.pretrained_resources == ("huggingface://facebook/w2v-bert-2.0",)
