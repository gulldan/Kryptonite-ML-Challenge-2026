import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


def load_audio(filepath: str, sample_rate: int) -> torch.Tensor:
    waveform, sr = sf.read(filepath, dtype="float32", always_2d=True)
    if waveform.shape[1] > 1:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    else:
        waveform = waveform[:, 0]
    waveform = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
    return waveform


def get_chunk(
    waveform: torch.Tensor | np.ndarray,
    chunk_len: tuple[int, int] | int,
    random_chunk: bool = True,
):
    """Get a fixed-length chunk.

    Args:
        waveform: torch.Tensor (1, samples) or (samples, )
        chunk_len: either a specific chunk length or a range within which the chunk length falls
        random_chunk: whether to take a random chunk. False takes a deterministic center chunk.
    Returns:
        torch.Tensor (1, exactly chunk_len) or (exactly chunk_len, )
    """
    ndim = len(waveform.shape)
    is_torch = False

    # first, convert to 1-dim numpy.array (to not confuse tile, repeat and so on)
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
        is_torch = True

    if ndim == 2:
        # squeeze it for now
        waveform = waveform[0]

    data_len = len(waveform)

    if isinstance(chunk_len, tuple):
        chunk_len = int(np.random.uniform(*chunk_len))
    if chunk_len <= 0:
        raise ValueError("chunk_len must be positive")

    if data_len >= chunk_len:
        if random_chunk:
            chunk_start = random.randint(0, data_len - chunk_len)
        else:
            chunk_start = (data_len - chunk_len) // 2
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


def get_eval_chunks(
    waveform: torch.Tensor | np.ndarray,
    chunk_len: int,
    num_chunks: int = 1,
) -> torch.Tensor | np.ndarray:
    """Return deterministic evenly spaced chunks for evaluation-time crop averaging."""

    if chunk_len <= 0:
        raise ValueError("chunk_len must be positive")
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")

    is_torch = isinstance(waveform, torch.Tensor)
    if is_torch:
        values = waveform.detach().cpu().numpy()
    else:
        values = np.asarray(waveform)
    if values.ndim == 2:
        values = values[0]

    data_len = len(values)
    if data_len >= chunk_len:
        max_start = data_len - chunk_len
        if num_chunks == 1:
            starts = np.asarray([max_start // 2], dtype=np.int64)
        else:
            starts = np.linspace(0, max_start, num=num_chunks).round().astype(np.int64)
        chunks = [values[int(start) : int(start) + chunk_len] for start in starts]
    elif data_len > 0:
        repeat_factor = chunk_len // data_len + 1
        padded = np.tile(values, repeat_factor)[:chunk_len]
        chunks = [padded for _ in range(num_chunks)]
    else:
        chunks = [np.zeros(chunk_len, dtype=np.float32) for _ in range(num_chunks)]

    result = np.stack(chunks, axis=0).astype(np.float32, copy=False)
    if is_torch:
        return torch.from_numpy(result)
    return result


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

        assert self.file_col in self.df.columns, f"Column '{self.file_col}' not found in CSV"
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
        if not Path(path).is_absolute() and self.base_dir:
            path = str(Path(self.base_dir) / path)

        waveform = load_audio(path, self.sample_rate)
        waveform = get_chunk(waveform, self.num_samples, self.is_train)
        if self.has_sid:
            spk = row[self.spk_col]
            label = self.speaker_to_label[spk]
        else:
            label = -1
        return waveform, label
