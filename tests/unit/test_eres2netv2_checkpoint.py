from __future__ import annotations

from pathlib import Path

import torch

from kryptonite.models.eres2netv2 import (
    ERes2NetV2Config,
    ERes2NetV2Encoder,
    load_eres2netv2_encoder_from_checkpoint,
    resolve_eres2netv2_checkpoint_path,
)


def test_load_eres2netv2_encoder_from_checkpoint_restores_model_state(tmp_path: Path) -> None:
    config = ERes2NetV2Config(
        feat_dim=8,
        embedding_size=16,
        m_channels=8,
        base_width=26,
        scale=2,
        expansion=2,
        num_blocks=(1, 1, 1, 1),
    )
    model = ERes2NetV2Encoder(config).to(device="cpu", dtype=torch.float32)
    checkpoint_root = tmp_path / "run"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_root / "eres2netv2_encoder.pt"
    torch.save(
        {
            "baseline_config": {
                "project": {},
                "data": {},
            },
            "model_config": {
                "feat_dim": config.feat_dim,
                "embedding_size": config.embedding_size,
                "m_channels": config.m_channels,
                "base_width": config.base_width,
                "scale": config.scale,
                "expansion": config.expansion,
                "num_blocks": list(config.num_blocks),
                "pooling_func": config.pooling_func,
                "two_embedding_layers": config.two_embedding_layers,
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    resolved_checkpoint, loaded_config, loaded_model = load_eres2netv2_encoder_from_checkpoint(
        torch=torch,
        checkpoint_path=checkpoint_root,
        project_root=tmp_path,
    )

    assert resolved_checkpoint == resolve_eres2netv2_checkpoint_path(
        checkpoint_path=checkpoint_root,
        project_root=tmp_path,
    )
    assert loaded_config.num_blocks == (1, 1, 1, 1)
    assert loaded_model.training is False
    torch.testing.assert_close(
        loaded_model.state_dict()["segment_1.weight"],
        model.state_dict()["segment_1.weight"],
    )
