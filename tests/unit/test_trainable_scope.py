from __future__ import annotations

import torch

from kryptonite.training.trainable_scope import (
    apply_encoder_trainable_scope,
    normalize_encoder_trainable_scope,
)


class TinyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 8, kernel_size=3)
        self.bn = torch.nn.BatchNorm1d(8)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(8, 6),
            torch.nn.BatchNorm1d(6),
        )


def test_batchnorm_affine_scope_freezes_non_batchnorm_parameters() -> None:
    encoder = TinyEncoder()

    summary = apply_encoder_trainable_scope(encoder, scope="bn-only")

    assert summary.scope == "batchnorm-affine"
    assert summary.batchnorm_module_count == 2
    assert encoder.conv.weight.requires_grad is False
    assert encoder.conv.bias is not None
    assert encoder.conv.bias.requires_grad is False
    assert encoder.proj[0].weight.requires_grad is False
    assert encoder.bn.weight.requires_grad is True
    assert encoder.bn.bias.requires_grad is True
    assert encoder.proj[1].weight.requires_grad is True
    assert encoder.proj[1].bias.requires_grad is True
    assert set(summary.trainable_tensors) == {
        "bn.weight",
        "bn.bias",
        "proj.1.weight",
        "proj.1.bias",
    }


def test_all_scope_unfreezes_every_encoder_parameter() -> None:
    encoder = TinyEncoder()
    apply_encoder_trainable_scope(encoder, scope="batchnorm-affine")

    summary = apply_encoder_trainable_scope(encoder, scope="all")

    assert summary.scope == "all"
    assert summary.trainable_parameters == summary.total_parameters
    assert all(parameter.requires_grad for parameter in encoder.parameters())


def test_normalize_encoder_trainable_scope_rejects_unknown_scope() -> None:
    try:
        normalize_encoder_trainable_scope("linear-only")
    except ValueError as error:
        assert "Unsupported encoder trainable scope" in str(error)
    else:
        raise AssertionError("expected ValueError")
