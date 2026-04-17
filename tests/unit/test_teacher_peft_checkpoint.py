from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

import kryptonite.training.teacher_peft.model as teacher_model
from kryptonite.training.teacher_peft import (
    TeacherPeftAdapterConfig,
    TeacherPeftEncoder,
    TeacherPeftModelConfig,
    load_teacher_peft_encoder_from_checkpoint,
    resolve_teacher_peft_checkpoint_path,
)


class FakeBackbone(torch.nn.Module):
    def __init__(self, hidden_size: int = 12) -> None:
        super().__init__()
        self.frame_projection = torch.nn.Linear(1, hidden_size)
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(
        self,
        *,
        input_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **_: object,
    ) -> SimpleNamespace:
        del attention_mask, output_hidden_states, return_dict
        assert input_values is not None
        hidden = self.frame_projection(input_values.unsqueeze(-1))
        return SimpleNamespace(last_hidden_state=hidden)


def test_resolve_teacher_peft_checkpoint_path_accepts_run_root_and_metadata_path(
    tmp_path: Path,
) -> None:
    checkpoint_dir = _write_teacher_checkpoint_fixture(tmp_path)

    assert (
        resolve_teacher_peft_checkpoint_path(
            checkpoint_path=tmp_path / "run",
            project_root=tmp_path,
        )
        == checkpoint_dir
    )
    assert (
        resolve_teacher_peft_checkpoint_path(
            checkpoint_path=checkpoint_dir / "checkpoint_metadata.json",
            project_root=tmp_path,
        )
        == checkpoint_dir
    )


def test_load_teacher_peft_encoder_from_checkpoint_restores_heads_and_trainable_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_dir = _write_teacher_checkpoint_fixture(tmp_path)
    calls: dict[str, object] = {}
    feature_extractor = object()

    def fake_auto_model_from_pretrained(
        model_id: str,
        *,
        revision: str | None,
        token: str | None,
    ) -> FakeBackbone:
        calls["model_id"] = model_id
        calls["revision"] = revision
        calls["token"] = token
        return FakeBackbone(hidden_size=12)

    class FakePeftModel:
        @staticmethod
        def from_pretrained(
            backbone: FakeBackbone,
            adapter_dir: Path,
            *,
            is_trainable: bool,
        ) -> FakeBackbone:
            calls["adapter_dir"] = Path(adapter_dir)
            calls["is_trainable"] = is_trainable
            for parameter in backbone.parameters():
                parameter.requires_grad = is_trainable
            return backbone

    def fake_feature_extractor_from_pretrained(
        checkpoint_path: Path,
        *,
        token: str | None,
    ) -> object:
        calls["feature_extractor_dir"] = Path(checkpoint_path)
        calls["feature_token"] = token
        return feature_extractor

    monkeypatch.setattr(
        teacher_model.AutoModel,
        "from_pretrained",
        fake_auto_model_from_pretrained,
    )
    monkeypatch.setattr(teacher_model, "PeftModel", FakePeftModel)
    monkeypatch.setattr(
        teacher_model.AutoFeatureExtractor,
        "from_pretrained",
        fake_feature_extractor_from_pretrained,
    )

    resolved_dir, metadata, returned_feature_extractor, encoder = (
        load_teacher_peft_encoder_from_checkpoint(
            checkpoint_path=checkpoint_dir,
            project_root=tmp_path,
            token="secret-token",
            trainable=True,
        )
    )

    assert resolved_dir == checkpoint_dir
    assert metadata["model"]["model_id"] == "fixture/wavlm"
    assert returned_feature_extractor is feature_extractor
    assert encoder.training is False
    assert calls["model_id"] == "fixture/wavlm"
    assert calls["revision"] == "main"
    assert calls["token"] == "secret-token"
    assert calls["adapter_dir"] == checkpoint_dir / "adapter"
    assert calls["feature_extractor_dir"] == checkpoint_dir / "feature_extractor"
    assert calls["feature_token"] == "secret-token"
    assert calls["is_trainable"] is True


def test_write_teacher_checkpoint_marks_merged_backbone_as_full_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_dir = tmp_path / "run" / "teacher_peft"

    class FakePeftModel(torch.nn.Module):
        pass

    class FakeMergedBackbone(FakeBackbone):
        def __init__(self, hidden_size: int = 12) -> None:
            super().__init__(hidden_size=hidden_size)
            self.peft_config = {"default": object()}

        def save_pretrained(self, output_dir: Path | str) -> None:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").write_text("{}", encoding="utf-8")
            (path / "model.safetensors").write_bytes(b"")

    class FakeFeatureExtractor:
        def save_pretrained(self, output_dir: Path | str) -> None:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(teacher_model, "PeftModel", FakePeftModel)

    encoder = TeacherPeftEncoder(
        backbone=FakeMergedBackbone(hidden_size=12),
        hidden_size=12,
        embedding_dim=8,
        projection_dropout=0.1,
    )
    classifier = torch.nn.Linear(8, 2)
    teacher_model.write_teacher_checkpoint(
        checkpoint_dir=checkpoint_dir,
        encoder=encoder,
        classifier=classifier,
        feature_extractor=FakeFeatureExtractor(),
        model_config=TeacherPeftModelConfig(
            model_id="fixture/wavlm",
            embedding_dim=8,
            projection_dropout=0.1,
        ),
        adapter_config=TeacherPeftAdapterConfig(
            rank=8,
            alpha=16,
            dropout=0.0,
            target_modules=("linear_q",),
        ),
        baseline_config={},
        speaker_to_index={"speaker_alpha": 0, "speaker_bravo": 1},
    )

    metadata = json.loads((checkpoint_dir / "checkpoint_metadata.json").read_text())
    assert metadata["checkpoint_format"] == "full_model"
    assert (checkpoint_dir / "adapter" / "config.json").is_file()
    assert (checkpoint_dir / "adapter" / "model.safetensors").is_file()


def _write_teacher_checkpoint_fixture(tmp_path: Path) -> Path:
    checkpoint_dir = tmp_path / "run" / "teacher_peft"
    (checkpoint_dir / "adapter").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "feature_extractor").mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "feature_extractor" / "preprocessor_config.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (checkpoint_dir / "adapter" / "adapter_config.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (checkpoint_dir / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "model": {
                    "model_id": "fixture/wavlm",
                    "revision": "main",
                    "embedding_dim": 8,
                },
                "baseline_config": {
                    "model": {
                        "projection_dropout": 0.1,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    template_encoder = TeacherPeftEncoder(
        backbone=FakeBackbone(hidden_size=12),
        hidden_size=12,
        embedding_dim=8,
        projection_dropout=0.1,
    )
    torch.save(
        {
            "encoder_head_state_dict": template_encoder.non_backbone_state_dict(),
        },
        checkpoint_dir / "heads.pt",
    )
    return checkpoint_dir
