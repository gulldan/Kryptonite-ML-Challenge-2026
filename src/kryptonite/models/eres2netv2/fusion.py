# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker).
# SPDX-License-Identifier: Apache-2.0

"""Adaptive feature-fusion blocks adapted from the Apache-licensed 3D-Speaker project."""

from __future__ import annotations

import torch
from torch import nn


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 4) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if reduction <= 0:
            raise ValueError("reduction must be positive")
        hidden_channels = max(1, channels // reduction)
        self.local_attention = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attention = self.local_attention(torch.cat((inputs, skip), dim=1))
        attention = 1.0 + torch.tanh(attention)
        return (inputs * attention) + (skip * (2.0 - attention))
