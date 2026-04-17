# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker).
# SPDX-License-Identifier: Apache-2.0

"""ERes2NetV2 speaker-embedding encoder adapted from the Apache-licensed 3D-Speaker project."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as torch_functional
from torch import nn

from .fusion import AdaptiveFeatureFusion
from .pooling import build_pooling_layer


class ClippedReLU(nn.Hardtanh):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(0.0, 20.0, inplace)


@dataclass(frozen=True, slots=True)
class ERes2NetV2Config:
    feat_dim: int = 80
    embedding_size: int = 192
    m_channels: int = 64
    base_width: int = 26
    scale: int = 2
    expansion: int = 2
    num_blocks: tuple[int, int, int, int] = (3, 4, 6, 3)
    pooling_func: str = "TSTP"
    two_embedding_layers: bool = False

    def __post_init__(self) -> None:
        if self.feat_dim <= 0:
            raise ValueError("feat_dim must be positive")
        if self.feat_dim < 8 or self.feat_dim % 8 != 0:
            raise ValueError("feat_dim must be a positive multiple of 8")
        if self.embedding_size <= 0:
            raise ValueError("embedding_size must be positive")
        if self.m_channels <= 0:
            raise ValueError("m_channels must be positive")
        if self.base_width <= 0:
            raise ValueError("base_width must be positive")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.expansion <= 0:
            raise ValueError("expansion must be positive")
        if len(self.num_blocks) != 4:
            raise ValueError("num_blocks must contain exactly four stage depths")
        if any(depth <= 0 for depth in self.num_blocks):
            raise ValueError("num_blocks values must be positive")


class ERes2NetV2Block(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        *,
        stride: int = 1,
        base_width: int = 26,
        scale: int = 2,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        width = int(math.floor(planes * (base_width / 64.0)))
        if width <= 0:
            raise ValueError("Computed block width must be positive")
        self.width = width
        self.scale = scale
        self.expansion = expansion
        self.conv1 = nn.Conv2d(
            in_planes,
            width * scale,
            kernel_size=1,
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.convs = nn.ModuleList(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False) for _ in range(scale)
        )
        self.batch_norms = nn.ModuleList(nn.BatchNorm2d(width) for _ in range(scale))
        self.relu = ClippedReLU(inplace=True)
        self.conv3 = nn.Conv2d(width * scale, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        if stride != 1 or in_planes != expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(expansion * planes),
            )
        else:
            self.shortcut = nn.Identity()

    def _split_and_process(self, outputs: torch.Tensor) -> torch.Tensor:
        branches = torch.split(outputs, self.width, dim=1)
        processed: list[torch.Tensor] = []
        for index in range(self.scale):
            if index == 0:
                branch = branches[index]
            else:
                branch = processed[-1] + branches[index]
            branch = self.convs[index](branch)
            branch = self.relu(self.batch_norms[index](branch))
            processed.append(branch)
        return torch.cat(processed, dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self._split_and_process(outputs)
        outputs = self.bn3(self.conv3(outputs))
        return self.relu(outputs + residual)


class ERes2NetV2AFFBlock(ERes2NetV2Block):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        *,
        stride: int = 1,
        base_width: int = 26,
        scale: int = 2,
        expansion: int = 2,
    ) -> None:
        super().__init__(
            in_planes,
            planes,
            stride=stride,
            base_width=base_width,
            scale=scale,
            expansion=expansion,
        )
        self.fusion_blocks = nn.ModuleList(
            AdaptiveFeatureFusion(self.width, reduction=4) for _ in range(scale - 1)
        )

    def _split_and_process(self, outputs: torch.Tensor) -> torch.Tensor:
        branches = torch.split(outputs, self.width, dim=1)
        processed: list[torch.Tensor] = []
        for index in range(self.scale):
            if index == 0:
                branch = branches[index]
            else:
                branch = self.fusion_blocks[index - 1](processed[-1], branches[index])
            branch = self.convs[index](branch)
            branch = self.relu(self.batch_norms[index](branch))
            processed.append(branch)
        return torch.cat(processed, dim=1)


class ERes2NetV2Encoder(nn.Module):
    def __init__(self, config: ERes2NetV2Config | None = None) -> None:
        super().__init__()
        self.config = config or ERes2NetV2Config()
        self.in_planes = self.config.m_channels
        self.stats_dim = int(self.config.feat_dim / 8) * self.config.m_channels * 8

        self.conv1 = nn.Conv2d(
            1,
            self.config.m_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.config.m_channels)
        self.layer1 = self._make_layer(
            ERes2NetV2Block,
            self.config.m_channels,
            self.config.num_blocks[0],
            stride=1,
        )
        self.layer2 = self._make_layer(
            ERes2NetV2Block,
            self.config.m_channels * 2,
            self.config.num_blocks[1],
            stride=2,
        )
        self.layer3 = self._make_layer(
            ERes2NetV2AFFBlock,
            self.config.m_channels * 4,
            self.config.num_blocks[2],
            stride=2,
        )
        self.layer4 = self._make_layer(
            ERes2NetV2AFFBlock,
            self.config.m_channels * 8,
            self.config.num_blocks[3],
            stride=2,
        )

        self.layer3_downsample = nn.Conv2d(
            self.config.m_channels * 4 * self.config.expansion,
            self.config.m_channels * 8 * self.config.expansion,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=False,
        )
        self.fuse34 = AdaptiveFeatureFusion(
            self.config.m_channels * 8 * self.config.expansion,
            reduction=4,
        )
        self.pool, n_stats = build_pooling_layer(self.config.pooling_func)
        self.segment_1 = nn.Linear(
            self.stats_dim * self.config.expansion * n_stats,
            self.config.embedding_size,
        )
        if self.config.two_embedding_layers:
            self.segment_batch_norm = nn.BatchNorm1d(self.config.embedding_size, affine=False)
            self.segment_2 = nn.Linear(self.config.embedding_size, self.config.embedding_size)
        else:
            self.segment_batch_norm = nn.Identity()
            self.segment_2 = nn.Identity()

    def _make_layer(
        self,
        block_cls: type[ERes2NetV2Block],
        planes: int,
        num_blocks: int,
        *,
        stride: int,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        for block_stride in [stride, *([1] * (num_blocks - 1))]:
            layers.append(
                block_cls(
                    self.in_planes,
                    planes,
                    stride=block_stride,
                    base_width=self.config.base_width,
                    scale=self.config.scale,
                    expansion=self.config.expansion,
                )
            )
            self.in_planes = planes * self.config.expansion
        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                "ERes2NetV2Encoder expects [batch, frames, feat_dim] input, "
                f"got {tuple(features.shape)}"
            )
        if features.shape[-1] != self.config.feat_dim:
            raise ValueError(
                "ERes2NetV2Encoder expects "
                f"feat_dim={self.config.feat_dim}, got {features.shape[-1]}"
            )

        outputs = features.permute(0, 2, 1).unsqueeze(1)
        outputs = torch_functional.relu(self.bn1(self.conv1(outputs)))
        layer1_outputs = self.layer1(outputs)
        layer2_outputs = self.layer2(layer1_outputs)
        layer3_outputs = self.layer3(layer2_outputs)
        layer4_outputs = self.layer4(layer3_outputs)
        layer3_downsampled = self.layer3_downsample(layer3_outputs)
        fused = self.fuse34(layer4_outputs, layer3_downsampled)
        stats = self.pool(fused)

        embedding_a = self.segment_1(stats)
        if not self.config.two_embedding_layers:
            return embedding_a

        normalized = torch_functional.relu(embedding_a)
        normalized = self.segment_batch_norm(normalized)
        return self.segment_2(normalized)
