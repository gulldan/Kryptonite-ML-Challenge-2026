# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker).
# SPDX-License-Identifier: Apache-2.0

"""Pooling layers adapted from the Apache-licensed 3D-Speaker project."""

from __future__ import annotations

import torch
from torch import nn


def _flatten_temporal_tensor(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.ndim == 4:
        return inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2], inputs.shape[3])
    if inputs.ndim == 3:
        return inputs
    raise ValueError(f"Expected a 3D or 4D tensor, got shape {tuple(inputs.shape)}")


class TemporalAveragePooling(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = _flatten_temporal_tensor(inputs)
        return flattened.mean(dim=-1)


class TemporalStdDevPooling(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = _flatten_temporal_tensor(inputs)
        return torch.sqrt(torch.var(flattened, dim=-1, unbiased=True) + 1e-8)


class TemporalStatisticsPooling(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = _flatten_temporal_tensor(inputs)
        mean = flattened.mean(dim=-1)
        std = torch.sqrt(torch.var(flattened, dim=-1, unbiased=True) + 1e-8)
        return torch.cat((mean, std), dim=1)


def build_pooling_layer(pooling_func: str) -> tuple[nn.Module, int]:
    normalized = pooling_func.upper()
    if normalized == "TAP":
        return TemporalAveragePooling(), 1
    if normalized == "TSDP":
        return TemporalStdDevPooling(), 1
    if normalized == "TSTP":
        return TemporalStatisticsPooling(), 2
    raise ValueError("pooling_func must be one of: TAP, TSDP, TSTP")
