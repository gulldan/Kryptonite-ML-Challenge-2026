"""CAM++ speaker-embedding encoder."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn

from .layers import (
    BasicResBlock,
    CAMDenseTDNNBlock,
    DenseLayer,
    StatisticsPool,
    TDNNLayer,
    TransitLayer,
    build_nonlinear,
)


@dataclass(frozen=True, slots=True)
class CAMPPlusConfig:
    feat_dim: int = 80
    embedding_size: int = 512
    growth_rate: int = 32
    bottleneck_scale: int = 4
    init_channels: int = 128
    head_channels: int = 32
    head_res_blocks: tuple[int, int] = (2, 2)
    tdnn_stride: int = 2
    block_layers: tuple[int, int, int] = (12, 24, 16)
    block_kernel_sizes: tuple[int, int, int] = (3, 3, 3)
    block_dilations: tuple[int, int, int] = (1, 2, 2)
    config: str = "batchnorm-relu"
    memory_efficient: bool = True

    def __post_init__(self) -> None:
        if self.feat_dim <= 0:
            raise ValueError("feat_dim must be positive")
        if self.feat_dim < 8 or self.feat_dim % 8 != 0:
            raise ValueError("feat_dim must be a multiple of 8 for the CAM++ front-end stem")
        if self.embedding_size <= 0:
            raise ValueError("embedding_size must be positive")
        if self.growth_rate <= 0:
            raise ValueError("growth_rate must be positive")
        if self.bottleneck_scale <= 0:
            raise ValueError("bottleneck_scale must be positive")
        if self.init_channels <= 0:
            raise ValueError("init_channels must be positive")
        if self.head_channels <= 0:
            raise ValueError("head_channels must be positive")
        if len(self.head_res_blocks) != 2:
            raise ValueError("head_res_blocks must contain exactly two stage depths")
        if any(depth <= 0 for depth in self.head_res_blocks):
            raise ValueError("head_res_blocks depths must be positive")
        if self.tdnn_stride <= 0:
            raise ValueError("tdnn_stride must be positive")
        if not (
            len(self.block_layers) == len(self.block_kernel_sizes) == len(self.block_dilations) == 3
        ):
            raise ValueError(
                "block_layers, block_kernel_sizes, and block_dilations must be length 3"
            )
        if any(depth <= 0 for depth in self.block_layers):
            raise ValueError("block_layers values must be positive")
        if any(kernel <= 0 or kernel % 2 == 0 for kernel in self.block_kernel_sizes):
            raise ValueError("block_kernel_sizes must contain positive odd values")
        if any(dilation <= 0 for dilation in self.block_dilations):
            raise ValueError("block_dilations must be positive")

    @property
    def bottleneck_channels(self) -> int:
        return self.bottleneck_scale * self.growth_rate


class FeatureContextModel(nn.Module):
    def __init__(
        self,
        *,
        feat_dim: int,
        channels: int,
        res_blocks: tuple[int, int],
    ) -> None:
        super().__init__()
        self.in_planes = channels
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer1 = self._make_layer(channels, res_blocks[0], stride=2)
        self.layer2 = self._make_layer(channels, res_blocks[1], stride=2)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=(2, 1),
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.out_channels = channels * (feat_dim // 8)

    def _make_layer(self, planes: int, num_blocks: int, *, stride: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for block_stride in [stride, *([1] * (num_blocks - 1))]:
            layers.append(BasicResBlock(self.in_planes, planes, stride=block_stride))
            self.in_planes = planes * BasicResBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs = features.unsqueeze(1)
        outputs = torch.relu(self.bn1(self.conv1(outputs)))
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = torch.relu(self.bn2(self.conv2(outputs)))
        batch, channels, freq, frames = outputs.shape
        return outputs.reshape(batch, channels * freq, frames)


class CAMPPlusEncoder(nn.Module):
    def __init__(self, config: CAMPPlusConfig | None = None) -> None:
        super().__init__()
        self.config = config or CAMPPlusConfig()
        self.head = FeatureContextModel(
            feat_dim=self.config.feat_dim,
            channels=self.config.head_channels,
            res_blocks=self.config.head_res_blocks,
        )
        channels = self.head.out_channels
        self.xvector = nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            self.config.init_channels,
                            5,
                            stride=self.config.tdnn_stride,
                            padding=-1,
                            dilation=1,
                            config=self.config.config,
                        ),
                    )
                ]
            )
        )
        channels = self.config.init_channels
        for index, (num_layers, kernel_size, dilation) in enumerate(
            zip(
                self.config.block_layers,
                self.config.block_kernel_sizes,
                self.config.block_dilations,
                strict=True,
            ),
            start=1,
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=self.config.growth_rate,
                bottleneck_channels=self.config.bottleneck_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                config=self.config.config,
                memory_efficient=self.config.memory_efficient,
            )
            self.xvector.add_module(f"block{index}", block)
            channels = channels + num_layers * self.config.growth_rate
            self.xvector.add_module(
                f"transit{index}",
                TransitLayer(channels, channels // 2, bias=False, config=self.config.config),
            )
            channels //= 2
        self.xvector.add_module("out_nonlinear", build_nonlinear(self.config.config, channels))
        self.xvector.add_module("stats", StatisticsPool())
        self.xvector.add_module(
            "dense",
            DenseLayer(channels * 2, self.config.embedding_size, config="batchnorm_"),
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                "CAMPPlusEncoder expects [batch, frames, feat_dim] input, "
                f"got {tuple(features.shape)}"
            )
        if features.shape[-1] != self.config.feat_dim:
            raise ValueError(
                f"CAMPPlusEncoder expects feat_dim={self.config.feat_dim}, got {features.shape[-1]}"
            )
        outputs = features.permute(0, 2, 1)
        outputs = self.head(outputs)
        outputs = self.xvector(outputs)
        return outputs
