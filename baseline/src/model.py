import torch
import torch.nn as nn

from typing import Union

from .mel_frontend import MelFrontend
from .ecapa import ECAPA_TDNN


class ECAPASpeakerId(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        embed_dim: int,
        num_classes: Union[None, int] = None,
    ):
        super().__init__()
        self.frontend = MelFrontend(sample_rate, n_fft, hop_length, n_mels)
        self.ecapa = ECAPA_TDNN(feat_dim=n_mels, embed_dim=embed_dim)

        self.classifier = None
        if num_classes:
            self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        feats = self.frontend(waveform)
        embeddings = self.ecapa(feats)

        if self.classifier:
            return self.classifier(embeddings)
        else:
            return embeddings

    def extract_embeddings(self, waveform: torch.Tensor) -> torch.Tensor:
        feats = self.frontend(waveform)
        embeddings = self.ecapa(feats)
        return embeddings
