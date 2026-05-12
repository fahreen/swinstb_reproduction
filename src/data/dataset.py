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
# PyTorch — the deep learning framework. Provides tensors (numpy arrays
# that can live on GPU), automatic differentiation, and neural network
# building blocks. Pretty much everything from here on uses torch.

from torch.utils.data import Dataset
# The base class for any PyTorch dataset. Inheriting from this and
# implementing __len__ and __getitem__ is enough to plug into DataLoader.


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
        # Sanity checks on the arguments
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
        # T and K are common shorthand in the paper. T = input length,
        # K = output (target) length. We store both as instance variables
        # so __getitem__ can use them.
        self.T = input_length
        self.K = target_length
        # Total window covers T input frames + K target frames.
        # For each training example we read this many frames consecutively.
        self.window = input_length + target_length
        self.stride = stride
        self.normalize = normalize

        # ─── Memory-map the cache ────────────────────────────────────────
        # mmap_mode='r' means:
        #   - 'r' = read-only (we won't modify the cache)
        #   - mmap = OS lazily reads disk blocks as we access them
        #
        # The returned object behaves like a normal numpy array but stays
        # mostly on disk. We only pay RAM cost for the slices we actually read.
        self._cache = np.load(cache_path, mmap_mode='r')

        # Verify shape and dtype match what preprocess_directory produced.
        # _cache.ndim is the number of dimensions (4 for shape (N, H, W, 3)).
        # _cache.shape[-1] is the last dimension's size (3 for RGB).
        if self._cache.ndim != 4 or self._cache.shape[-1] != 3:
            raise ValueError(
                f"Expected cache shape (N, H, W, 3), got {self._cache.shape}"
            )
        if self._cache.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 cache, got dtype {self._cache.dtype}"
            )

        # len() on a numpy array returns its size along axis 0.
        # For our (N, H, W, 3) array, this is N (the number of frames).
        n_total = len(self._cache)

        # ─── Compute chronological split boundaries ──────────────────────
        # split_ratio = (4, 1, 1) means train:val:test = 4:1:1 in proportion.
        # Total parts = 4 + 1 + 1 = 6.
        # Train gets first 4/6 ≈ 67% of frames.
        # Val gets next 1/6 ≈ 17%.
        # Test gets last 1/6 ≈ 17%.
        r_train, r_val, r_test = split_ratio
        r_sum = r_train + r_val + r_test

        # Integer division (//) ensures we get a clean integer index.
        # For n_total = 10777 and r_train/r_sum = 4/6:
        #   train_end = (10777 * 4) // 6 = 7184
        # So train spans frames 0..7183, val 7184..8980, test 8981..10776.
        train_end = (n_total * r_train) // r_sum
        val_end = (n_total * (r_train + r_val)) // r_sum

        # Pick the right boundaries for this split.
        if split == 'train':
            self.split_start, self.split_end = 0, train_end
        elif split == 'val':
            self.split_start, self.split_end = train_end, val_end
        else:  # 'test'
            self.split_start, self.split_end = val_end, n_total

        # Verify the split has enough frames for at least one window.
        # Critical check: with input=20 and target=20, we need at least 40
        # frames in this split to produce one example.
        split_size = self.split_end - self.split_start
        if split_size < self.window:
            raise ValueError(
                f"Split '{split}' has {split_size} frames but window of "
                f"{self.window} frames is needed for one example."
            )

        # ─── Count valid sliding windows ─────────────────────────────────
        # Sliding window: starts at split_start, slides by `stride` each step,
        # last valid start is when the window's END is still within the split.
        #
        # If split_size = 7184 and window = 40 and stride = 1:
        #   First window: frames 0..39
        #   Last window:  frames 7144..7183 (start at 7144)
        #   Number of windows: 7184 - 40 + 1 = 7145
        #
        # Formula generalized for stride s:
        #   n_examples = (split_size - window) // stride + 1
        #
        # With stride=1, this gives the maximum number of overlapping windows.
        # Each consecutive example shares window-1 = 39 frames with the previous.
        self.n_examples = (split_size - self.window) // stride + 1

    def __len__(self) -> int:
        # PyTorch's DataLoader calls len(dataset) to know how many examples to
        # iterate over.
        return self.n_examples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # __getitem__ is called by DataLoader to fetch each example.
        # If you do `dataset[5]`, Python calls dataset.__getitem__(5).
        # DataLoader calls this many times in parallel via num_workers
        # subprocesses, then collates results into batches.

        if idx < 0 or idx >= self.n_examples:
            raise IndexError(
                f"Index {idx} out of range for {len(self)} examples in "
                f"split '{self.split}'"
            )

        # ─── Locate the source slice ─────────────────────────────────────
        # Compute where in the FULL cache this example's window starts.
        # Example idx=0 in train split: start = 0 + 0*1 = 0, end = 40.
        # Example idx=5 in train split: start = 0 + 5*1 = 5, end = 45.
        # Example idx=0 in val split:   start = 7184 + 0 = 7184, end = 7224.
        start = self.split_start + idx * self.stride
        end = start + self.window

        # ─── Read the slice from mmap ────────────────────────────────────
        # self._cache[start:end] returns a VIEW into the mmap'd file —
        # this is the slice we want, but it's still backed by disk.
        #
        # np.array(...) FORCES a copy into a real in-memory numpy array.
        # Why copy? Because:
        # 1. The mmap might be invalidated later if multiple workers access it.
        # 2. We need to do dtype conversion and reshaping, which can't be
        #    done lazily on a view.
        # 3. The copy is only ~7 MB (40 frames × 256 × 256 × 3 bytes),
        #    so the cost is negligible.
        seq = np.array(self._cache[start:end])  # (T+K, H, W, 3) uint8

        # ─── Convert uint8 → float32 ─────────────────────────────────────
        # The model expects float inputs in [0, 1].
        # Dividing a uint8 by 255 NumPy converts to float and rescales.
        # Resulting values are 0.0 to 1.0 (approximately, since 255/255 = 1.0
        # but pixel value 254 becomes 0.996, etc.).
        #
        # Why .astype(np.float32) first? Because doing /255 directly on uint8
        # would do integer division. We need float math.
        if self.normalize:
            seq_f = seq.astype(np.float32) / 255.0
        else:
            seq_f = seq.astype(np.float32)

        # ─── Convert NumPy → PyTorch tensor and permute axes ─────────────
        #
        # torch.from_numpy(seq_f) creates a torch.Tensor that SHARES MEMORY
        # with the numpy array. No copy. Same data, two interfaces.
        #
        # .permute(3, 0, 1, 2) rearranges axes:
        #   Input  axes order: (T+K, H, W, 3)   ← indices 0, 1, 2, 3
        #   permute(3, 0, 1, 2) → axes (3, 0, 1, 2) → (3, T+K, H, W)
        #
        # Result is channels-first, which is what PyTorch convs expect.
        # The new shape (3, T+K, H, W) is the channels-first 3D video format.
        #
        # .contiguous() forces the tensor to use contiguous memory.
        # Permute creates a "view" with non-contiguous strides; some PyTorch
        # ops require contiguous data, so we make it so explicitly.
        # Slight memory copy, but standard practice after permute().
        seq_t = torch.from_numpy(seq_f).permute(3, 0, 1, 2).contiguous()

        # ─── Split into input and target ─────────────────────────────────
        # seq_t has shape (3, 40, 256, 256) for T=K=20.
        # First 20 time-steps along axis 1 → input.
        # Last 20 time-steps along axis 1 → target.
        #
        # Tensor slicing syntax: tensor[:, :self.T] means:
        #   - Axis 0 (channels): take all 3 with ":"
        #   - Axis 1 (time): take indices 0..T-1 with ":self.T"
        #   - Axes 2 and 3 (H, W): not specified, default to all
        input_seq = seq_t[:, : self.T]
        # input_seq shape: (3, 20, 256, 256)

        target_seq = seq_t[:, self.T :]
        # target_seq shape: (3, 20, 256, 256)
        # Frames 20-39 of the original sequence (the future).

        # Return as tuple. PyTorch's DataLoader will collate many of these
        # into a batch by adding a new leading dimension:
        # input batch shape: (batch_size, 3, 20, 256, 256)
        return input_seq, target_seq

    def describe(self) -> str:
        """Return a human-readable summary of this split."""
        # Diagnostic helper — used by scripts to log dataset sizes.
        # Not part of the PyTorch interface; just convenience.
        split_size = self.split_end - self.split_start
        return (
            f"SpectrogramSequenceDataset(split='{self.split}'): "
            f"{self.n_examples} examples from {split_size} source frames "
            f"(indices {self.split_start}..{self.split_end-1}), "
            f"window=T+K={self.window}, stride={self.stride}"
        )