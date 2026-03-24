"""Classification heads and margin loss for CAM++ baseline training."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as torch_functional
from torch import nn

from .layers import DenseLayer


class CosineClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        num_classes: int,
        num_blocks: int = 0,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.blocks = nn.ModuleList()
        current_dim = input_dim
        for _ in range(num_blocks):
            self.blocks.append(DenseLayer(current_dim, hidden_dim, config="batchnorm"))
            current_dim = hidden_dim
        self.weight = nn.Parameter(torch.empty(num_classes, current_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        outputs = embeddings
        for layer in self.blocks:
            outputs = layer(outputs)
        return torch_functional.linear(
            torch_functional.normalize(outputs),
            torch_functional.normalize(self.weight),
        )


class ArcMarginLoss(nn.Module):
    def __init__(
        self,
        *,
        scale: float = 32.0,
        margin: float = 0.2,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        if margin < 0.0:
            raise ValueError("margin must be non-negative")
        self.scale = scale
        self.easy_margin = easy_margin
        self.criterion = nn.CrossEntropyLoss()
        self.update(margin)

    def update(self, margin: float = 0.2) -> None:
        self.margin = margin
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.margin_adjustment = 1.0 + math.cos(math.pi - margin)

    def forward(self, cosine_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sine = torch.sqrt((1.0 - cosine_logits.pow(2)).clamp_min(1e-7))
        phi = cosine_logits * self.cos_margin - sine * self.sin_margin
        if self.easy_margin:
            phi = torch.where(cosine_logits > 0.0, phi, cosine_logits)
        else:
            phi = torch.where(
                cosine_logits > self.threshold,
                phi,
                cosine_logits - self.margin_adjustment,
            )

        one_hot = torch.zeros_like(cosine_logits)
        one_hot.scatter_(1, labels.unsqueeze(1).long(), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine_logits)
        return self.criterion(output * self.scale, labels)
