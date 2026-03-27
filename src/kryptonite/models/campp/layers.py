"""CAM++ building blocks adapted from the Apache-licensed 3D-Speaker project."""

from __future__ import annotations

import torch
import torch.nn.functional as torch_functional
import torch.utils.checkpoint as checkpoint
from torch import nn


def build_nonlinear(config: str, channels: int) -> nn.Sequential:
    layer = nn.Sequential()
    for name in config.split("-"):
        if name == "relu":
            layer.add_module("relu", nn.ReLU(inplace=True))
            continue
        if name == "prelu":
            layer.add_module("prelu", nn.PReLU(channels))
            continue
        if name == "batchnorm":
            layer.add_module("batchnorm", nn.BatchNorm1d(channels))
            continue
        if name == "batchnorm_":
            layer.add_module("batchnorm", nn.BatchNorm1d(channels, affine=False))
            continue
        raise ValueError(f"Unexpected nonlinear block component {name!r}.")
    return layer


def statistics_pooling(
    inputs: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
    unbiased: bool = True,
) -> torch.Tensor:
    mean = inputs.mean(dim=dim)
    std = inputs.std(dim=dim, unbiased=unbiased)
    pooled = torch.cat([mean, std], dim=-1)
    if keepdim:
        pooled = pooled.unsqueeze(dim=dim)
    return pooled


class StatisticsPool(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return statistics_pooling(inputs)


class TDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        config: str = "batchnorm-relu",
    ) -> None:
        super().__init__()
        if padding < 0:
            if kernel_size % 2 == 0:
                raise ValueError(f"Expected odd kernel_size for auto-padding, got {kernel_size}.")
            padding = ((kernel_size - 1) // 2) * dilation
        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = build_nonlinear(config, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(self.linear(inputs))


class ContextAwareMaskingLayer(nn.Module):
    def __init__(
        self,
        bottleneck_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int,
        padding: int,
        dilation: int,
        bias: bool,
        reduction: int = 2,
    ) -> None:
        super().__init__()
        reduced_channels = max(1, bottleneck_channels // reduction)
        self.local = nn.Conv1d(
            bottleneck_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.context_down = nn.Conv1d(bottleneck_channels, reduced_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.context_up = nn.Conv1d(reduced_channels, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        local = self.local(inputs)
        context = inputs.mean(dim=-1, keepdim=True) + self.segment_pooling(inputs)
        context = self.relu(self.context_down(context))
        mask = self.sigmoid(self.context_up(context))
        return local * mask

    def segment_pooling(self, inputs: torch.Tensor, *, segment_length: int = 100) -> torch.Tensor:
        pooled = torch_functional.avg_pool1d(
            inputs,
            kernel_size=segment_length,
            stride=segment_length,
            ceil_mode=True,
        )
        # Gather each frame from its segment-average index instead of reshaping an
        # expanded tensor. This stays symbolically shape-safe for torch.export.
        frame_indices = torch.arange(inputs.shape[-1], device=inputs.device)
        segment_indices = torch.div(frame_indices, segment_length, rounding_mode="floor")
        return pooled.index_select(-1, segment_indices)


class CAMDenseTDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config: str = "batchnorm-relu",
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"Expected odd kernel_size, got {kernel_size}.")
        padding = ((kernel_size - 1) // 2) * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear_in = build_nonlinear(config, in_channels)
        self.project = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
        self.nonlinear_bottleneck = build_nonlinear(config, bottleneck_channels)
        self.cam = ContextAwareMaskingLayer(
            bottleneck_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def _project(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.project(self.nonlinear_in(inputs))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training and self.memory_efficient:
            projected = checkpoint.checkpoint(self._project, inputs, use_reentrant=False)
        else:
            projected = self._project(inputs)
        return self.cam(self.nonlinear_bottleneck(projected))


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(
        self,
        *,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config: str = "batchnorm-relu",
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for index in range(num_layers):
            self.add_module(
                f"tdnnd{index + 1}",
                CAMDenseTDNNLayer(
                    in_channels=in_channels + index * out_channels,
                    out_channels=out_channels,
                    bottleneck_channels=bottleneck_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    bias=bias,
                    config=config,
                    memory_efficient=memory_efficient,
                ),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for layer in self:
            outputs = torch.cat([outputs, layer(outputs)], dim=1)
        return outputs


class TransitLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bias: bool = True,
        config: str = "batchnorm-relu",
    ) -> None:
        super().__init__()
        self.nonlinear = build_nonlinear(config, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.nonlinear(inputs))


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bias: bool = False,
        config: str = "batchnorm-relu",
    ) -> None:
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = build_nonlinear(config, out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            projected = self.linear(inputs.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            projected = self.linear(inputs)
        return self.nonlinear(projected)


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=(stride, 1),
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch_functional.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        outputs = outputs + self.shortcut(inputs)
        return torch_functional.relu(outputs)
