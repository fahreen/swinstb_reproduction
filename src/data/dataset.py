"""
PyTorch Dataset for serving (input, target) sequence pairs to the training loop.

Each example is a pair of consecutive sub-sequences extracted from the
preprocessed spectrogram array:

    input  : T consecutive frames, used to predict the future
    target : K consecutive frames immediately following the input

For the paper's default config: T = K = 20.

Data layout:
    The cache file is shape (N, H, W, 3) uint8, chronologically ordered.
    A sliding window of length T+K is moved across this array. Each
    valid window position produces one training example.

Channel convention:
    The cache stores frames as (H, W, 3) — channels-last, NumPy/PIL convention.
    PyTorch convolutional layers expect channels-first. We permute to
    (3, T, H, W) — channels-first, with the time axis between channels and
    spatial dimensions. This is what `MONAI.SwinUNETR` expects for 3D input
    (it treats the time axis as a "depth" dimension).

Splits:
    Chronological 4:1:1 (train:val:test) per Pan et al. Section VI-A.
    Splits are non-overlapping in source frames — but within a split,
    sliding windows can overlap (consecutive examples share T+K-1 frames).

Memory:
    The cache is opened with `mmap_mode='r'`, so only the bytes needed for
    each example are loaded. The model and one batch comfortably fit in
    GPU memory; the host doesn't need to hold the full 2 GB array.
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrogramSequenceDataset(Dataset):
    """
    Sliding-window dataset over a preprocessed spectrogram cache.

    Args:
        cache_path: Path to the .npy file produced by preprocess_directory.
        split: One of 'train', 'val', 'test'.
        input_length: T — number of input frames per example.
        target_length: K — number of target frames per example.
        split_ratio: Tuple of three ints. Default (4, 1, 1) per paper.
        stride: Step between sliding-window starts. 1 = maximum coverage
            (overlapping examples). Larger values give fewer, faster epochs.
        normalize: If True (default), divide pixel values by 255 to yield
            float32 in [0, 1]. If False, keep raw uint8 (rarely useful).

    Yields per __getitem__:
        Tuple of two torch tensors:
            input:  shape (3, T, H, W), dtype float32
            target: shape (3, K, H, W), dtype float32

    Example:
        >>> ds = SpectrogramSequenceDataset('fm_spectrograms.npy', 'train')
        >>> len(ds)
        7165
        >>> x, y = ds[0]
        >>> x.shape, y.shape
        (torch.Size([3, 20, 256, 256]), torch.Size([3, 20, 256, 256]))
    """

    def __init__(
        self,
        cache_path: str,
        split: str = 'train',
        input_length: int = 20,
        target_length: int = 20,
        split_ratio: Tuple[int, int, int] = (4, 1, 1),
        stride: int = 1,
        normalize: bool = True,
    ):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be 'train'/'val'/'test', got {split!r}")
        if input_length < 1 or target_length < 1:
            raise ValueError("input_length and target_length must be >= 1")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if len(split_ratio) != 3 or any(r <= 0 for r in split_ratio):
            raise ValueError(
                f"split_ratio must be three positive ints, got {split_ratio}"
            )

        self.cache_path = cache_path
        self.split = split
        self.T = input_length
        self.K = target_length
        self.window = input_length + target_length
        self.stride = stride
        self.normalize = normalize

        # Memory-map the cache so we don't load 2 GB into RAM
        self._cache = np.load(cache_path, mmap_mode='r')
        if self._cache.ndim != 4 or self._cache.shape[-1] != 3:
            raise ValueError(
                f"Expected cache shape (N, H, W, 3), got {self._cache.shape}"
            )
        if self._cache.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 cache, got dtype {self._cache.dtype}"
            )

        n_total = len(self._cache)

        # Compute chronological split boundaries
        r_train, r_val, r_test = split_ratio
        r_sum = r_train + r_val + r_test
        train_end = (n_total * r_train) // r_sum
        val_end = (n_total * (r_train + r_val)) // r_sum

        if split == 'train':
            self.split_start, self.split_end = 0, train_end
        elif split == 'val':
            self.split_start, self.split_end = train_end, val_end
        else:  # 'test'
            self.split_start, self.split_end = val_end, n_total

        split_size = self.split_end - self.split_start
        if split_size < self.window:
            raise ValueError(
                f"Split '{split}' has {split_size} frames but window of "
                f"{self.window} frames is needed for one example."
            )

        # Number of valid sliding windows
        self.n_examples = (split_size - self.window) // stride + 1

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.n_examples:
            raise IndexError(
                f"Index {idx} out of range for {len(self)} examples in "
                f"split '{self.split}'"
            )

        # Compute the absolute starting frame in the cache
        start = self.split_start + idx * self.stride
        end = start + self.window

        # Slice from mmap (this only reads the bytes we need)
        # Copy out so we don't hold a view into the mmap forever.
        seq = np.array(self._cache[start:end])  # (T+K, H, W, 3) uint8

        # Convert to float32 in [0, 1]
        if self.normalize:
            seq_f = seq.astype(np.float32) / 255.0
        else:
            seq_f = seq.astype(np.float32)

        # Permute (T+K, H, W, 3) → (3, T+K, H, W)
        seq_t = torch.from_numpy(seq_f).permute(3, 0, 1, 2).contiguous()

        # Split input vs target
        input_seq = seq_t[:, : self.T]
        target_seq = seq_t[:, self.T :]

        return input_seq, target_seq

    def describe(self) -> str:
        """Return a human-readable summary of this split."""
        split_size = self.split_end - self.split_start
        return (
            f"SpectrogramSequenceDataset(split='{self.split}'): "
            f"{self.n_examples} examples from {split_size} source frames "
            f"(indices {self.split_start}..{self.split_end-1}), "
            f"window=T+K={self.window}, stride={self.stride}"
        )