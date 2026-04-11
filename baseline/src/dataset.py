import os
import random
from typing import Tuple, Union
import numpy as np
import pandas as pd
import torch
import torchaudio

from torch.utils.data import Dataset


def load_audio(filepath: str, sample_rate: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze(0)


def get_chunk(
    waveform: Union[torch.Tensor, np.ndarray],
    chunk_len: Union[Tuple[int, int], int],
    random_chunk: bool = True,
):
    """Get random chunk

    Args:
        waveform: torch.Tensor (1, samples) or (samples, )
        chunk_len: either a specific chunk length or a range within which the chunk length falls
        random_chunk: whether to take a random chunk or not
    Returns:
        torch.Tensor (1, exactly chunk_len) or (exactly chunk_len, )
    """
    ndim = len(waveform.shape)
    is_torch = False

    # first, convert to 1-dim numpy.array (to not confuse tile, repeat and so on)
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().numpy()
        is_torch = True

    if ndim == 2:
        # squeeze it for now
        waveform = waveform[0]

    data_len = len(waveform)

    if isinstance(chunk_len, Tuple):
        chunk_len = int(np.random.uniform(*chunk_len))

    if data_len >= chunk_len:
        chunk_start = 0
        if random_chunk:
            chunk_start = random.randint(0, data_len - chunk_len)
        waveform = waveform[chunk_start : chunk_start + chunk_len]

    elif data_len > 0:
        repeat_factor = chunk_len // data_len + 1
        waveform = np.tile(waveform, repeat_factor)
        waveform = waveform[:chunk_len]
    else:
        print("Trying to pad an audio of zero length.")
        waveform = np.zeros(chunk_len, dtype=np.float32)

    if ndim == 2:
        waveform = waveform[None, :]

    if is_torch:
        waveform = torch.tensor(waveform)

    return waveform


class SpeakerDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        sample_rate: int,
        chunk_seconds: float,
        is_train: bool,
        base_dir: str | None = None,
        filepath_col: str = "filepath",
        speaker_id_col: str = "speaker_id",
    ):
        self.df = pd.read_csv(csv_path)
        self.file_col = filepath_col
        self.spk_col = speaker_id_col
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * float(chunk_seconds))
        self.is_train = is_train
        self.base_dir = base_dir

        assert (
            self.file_col in self.df.columns
        ), f"Column '{self.file_col}' not found in CSV"
        self.has_sid = self.spk_col in self.df.columns

        if self.has_sid:
            speakers = sorted(self.df[self.spk_col].unique())
            self.speaker_to_label = {spk: i for i, spk in enumerate(speakers)}
        else:
            self.speaker_to_label = {}
        self.num_speakers = len(self.speaker_to_label)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row[self.file_col]
        if not os.path.isabs(path) and self.base_dir:
            path = os.path.normpath(os.path.join(self.base_dir, path))

        waveform = load_audio(path, self.sample_rate)
        waveform = get_chunk(waveform, self.num_samples, not self.is_train)
        if self.has_sid:
            spk = row[self.spk_col]
            label = self.speaker_to_label[spk]
        else:
            label = -1
        return waveform, label
