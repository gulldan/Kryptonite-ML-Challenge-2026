import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class FilterbankFeatures(nn.Module):
    """
    ONNX log-mel filterbank features (conv1d -> power -> mel -> log).
    """

    LOG_ZERO_GUARD_VALUE = 2**-24

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        preemphasis_coefficient: float = 0.0,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.win_length = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.n_mels = int(n_mels)

        window_tensor = torch.hann_window(self.win_length, periodic=False)
        forward_basis = self._compute_forward_basis(
            self.n_fft, window_tensor, float(preemphasis_coefficient)
        )
        self.register_buffer("forward_basis", forward_basis, persistent=False)

        filterbanks = self._compute_filterbanks(
            self.n_fft, self.sample_rate, self.n_mels
        )
        self.register_buffer("filterbanks", filterbanks, persistent=False)

    @staticmethod
    def _compute_forward_basis(
        n_fft: int, window: torch.Tensor, preemphasis_coefficient: float
    ) -> torch.Tensor:
        window_size = int(window.size(-1))
        eye = torch.eye(n_fft, dtype=torch.float64)
        fourier_basis = torch.fft.fft(eye, norm="backward")
        fourier_basis = fourier_basis[: n_fft // 2 + 1]
        forward_basis = torch.cat((fourier_basis.real, fourier_basis.imag), dim=0).T
        forward_basis *= window[:, None].to(forward_basis.dtype)

        if preemphasis_coefficient != 0.0:
            preemphasis_matrix = forward_basis.new_ones(window_size).diag(0)
            preemphasis_matrix -= forward_basis.new_full(
                (window_size - 1,), preemphasis_coefficient
            ).diag(1)
            preemphasis_matrix[0, 0] -= preemphasis_coefficient
            forward_basis = preemphasis_matrix @ forward_basis

        return forward_basis.T[:, None, :].float().contiguous()

    @staticmethod
    def _compute_filterbanks(n_fft: int, sample_rate: int, n_mels: int) -> torch.Tensor:
        filterbanks = torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=float(sample_rate) / 2.0,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        return filterbanks.T.contiguous()  # (n_mels, n_freqs)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        x = waveform[:, None].float()  # (B, 1, T)
        spectrum = F.conv1d(x, self.forward_basis.float(), stride=self.hop_length)
        spectrum = spectrum.view(
            spectrum.size(0), 2, self.forward_basis.size(0) // 2, -1
        )
        power = spectrum.square().sum(dim=1)  # (B, n_freqs, frames)
        mel = self.filterbanks.to(power.dtype) @ power  # (B, n_mels, frames)
        mel = torch.log(mel + float(self.LOG_ZERO_GUARD_VALUE))
        return mel.permute(0, 2, 1)  # (B, frames, n_mels)


class MelFrontend(nn.Module):
    """
    Unified frontend used in training/inference/export.
    """

    def __init__(self, sample_rate: int, n_fft: int, hop_length: int, n_mels: int):
        super().__init__()
        self.fbank = FilterbankFeatures(sample_rate, n_fft, hop_length, n_mels)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.fbank(waveform)
