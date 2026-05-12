"""
Batch preprocessing: convert a directory of raw .xls files into a single
cached numpy array of RGB spectrograms.

Why a single .npy file:
    - Loading is fast (memory-mapped or one-shot).
    - The whole FM dataset (~2 GB at 256x256x3 uint8) fits comfortably in RAM.
    - Training reads one file, not 10,777.

Multiprocessing:
    Each .xls file is independent — no joins, no aggregation across files.
    Embarrassingly parallel. We use multiprocessing.Pool with N workers
    (default = os.cpu_count() - 1) to parallelise the per-file work.

Memory bookkeeping:
    Output array shape: (N, 256, 256, 3) uint8.
    For FM (N=10,777): N * 256 * 256 * 3 = ~2.0 GB.
    For LTE (N≈2,400): ~480 MB.

    During preprocessing we hold the full output array in RAM and write
    workers' results into it at known indices. This is simpler and faster
    than streaming-to-disk for datasets of this size.
"""

import os
import multiprocessing as mp
# Python's standard library for running tasks in parallel processes.
# Each process gets its own Python interpreter, so they don't share the GIL
# (Global Interpreter Lock) that would otherwise prevent true parallelism.
# Costs: process startup time + serialization overhead for data passed
# between processes. For long-running CPU-bound work (like STFT here),
# the speedup is roughly proportional to the number of CPU cores.

from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm
# tqdm = "progress bar" library. Wrap any iterable with tqdm() and it
# shows a live progress bar in the terminal: 45%|████▌     | 4500/10000

from .parse_iq import parse_iq_file, list_files_chronological
from .stft import iq_to_spectrogram


# ─────────────────────────────────────────────────────────────────────────────
# Worker function
# ─────────────────────────────────────────────────────────────────────────────

def _process_one_file(args: Tuple[int, str]) -> Tuple[int, np.ndarray]:
    """
    Worker function: parse one file and convert to spectrogram.

    Designed to be called by multiprocessing.Pool. Returns the file's
    index and its RGB spectrogram so the main process can assemble the
    output array in order.

    Args:
        args: tuple of (index, filepath)

    Returns:
        (index, rgb_spectrogram) where rgb_spectrogram is uint8 (256,256,3)

    Raises:
        Exception: if parsing or STFT fails for this file. The Pool will
            propagate this back to the main process.
    """
    # This function runs in a SEPARATE PROCESS — not the main one.
    # Python's multiprocessing pickles the function arguments, sends them
    # to a worker process, runs the function, and pickles the return value
    # back to the main process. The 1-argument tuple form is convenient
    # for pool.imap_unordered() which takes single-argument functions.
    idx, filepath = args

    # Two-step pipeline per file:
    # 1. parse_iq_file → 1D complex array of I/Q samples (~32508 values)
    # 2. iq_to_spectrogram → (256, 256, 3) uint8 RGB image
    iq = parse_iq_file(filepath)
    rgb = iq_to_spectrogram(iq)

    # We return BOTH the index AND the result, because workers may finish
    # out of order (imap_unordered). The main process uses idx to place
    # this result in the correct slot of the output array.
    return idx, rgb


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_directory(
    input_dir: str,
    output_path: str,
    num_workers: Optional[int] = None,
    output_size: int = 256,
    overwrite: bool = False,
) -> dict:
    """
    Process all .xls files in input_dir to a consolidated .npy file.

    Args:
        input_dir: Directory containing NUAA .xls files.
        output_path: Where to save the consolidated array (e.g.
            './data/processed/fm_spectrograms.npy').
        num_workers: Number of parallel workers. Default = cpu_count() - 1.
        output_size: Spatial resolution of each spectrogram (default 256).
        overwrite: If False (default), refuse to overwrite an existing
            output file. If True, replace it.

    Returns:
        Dict with keys:
            'n_files':      number of files processed
            'output_shape': shape of saved array (N, H, W, 3)
            'output_path':  path to saved file
            'failed':       list of (filepath, error_message) for any failures

    Raises:
        FileExistsError: if output exists and overwrite=False.
        ValueError: if input directory has no .xls files.
    """
    # Sanity checks
    if os.path.exists(output_path) and not overwrite:
        # Defensive: a typical full run takes minutes; we don't want to
        # silently overwrite hours of previously-processed data.
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            f"Pass overwrite=True to replace it."
        )

    # CRITICAL: chronological listing.
    # list_files_chronological() sorts by HHMMSS timestamp.
    # The downstream code (dataset.py) will rely on this ordering to do
    # the 4:1:1 chronological train/val/test split. If files were in
    # arbitrary order, the temporal structure would be lost.
    files = list_files_chronological(input_dir)
    if not files:
        raise ValueError(f"No .xls files found in {input_dir}")

    n_files = len(files)
    print(f"Found {n_files} files in {input_dir}")
    print(f"Output: {output_path}")

    # ─── Allocate output array up front ───────────────────────────────────
    #
    # np.zeros allocates AND ZERO-INITIALIZES.
    # We use zeros (instead of np.empty) here as a safety measure: if any
    # worker crashes silently and a slot isn't filled, that slot stays
    # zero rather than containing garbage. Easier to detect failures.
    #
    # Shape: (n_files, 256, 256, 3) — one slot per file.
    # dtype=np.uint8 — each pixel is 0-255 (1 byte per channel).
    #
    # Memory: 10777 * 256 * 256 * 3 * 1 byte = 2,123,366,400 bytes ≈ 1.98 GB.
    # nbytes is a numpy property giving exact byte count.
    output = np.zeros((n_files, output_size, output_size, 3), dtype=np.uint8)
    print(f"Allocated output array: shape {output.shape} "
          f"(~{output.nbytes / 1e9:.2f} GB)")

    # ─── Decide worker count ──────────────────────────────────────────────
    if num_workers is None:
        # Default: use all CPU cores minus one.
        # The "minus one" leaves a core free for the OS, the main process,
        # and any other tasks the user might be doing.
        # The "max(1, ...)" guards against systems where cpu_count() returns 1.
        num_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Using {num_workers} worker process{'es' if num_workers != 1 else ''}")
    print()

    # ─── Build work items ─────────────────────────────────────────────────
    # enumerate(files) yields (0, file0), (1, file1), (2, file2), ...
    # We materialize this as a list because we need the same iterable later
    # and need to know its length.
    work_items = list(enumerate(files))

    # Track failures so we can report them at the end
    failed: List[Tuple[str, str]] = []

    # ─── Execute work, single-process vs multi-process ────────────────────
    if num_workers == 1:
        # Single-process path: easier to debug because errors raise normally
        # (in multiprocessing, exceptions in workers can be harder to trace).
        for idx, filepath in tqdm(work_items, total=n_files, desc='Processing'):
            try:
                _, rgb = _process_one_file((idx, filepath))
                output[idx] = rgb
                # Note: output[idx] is the entire (256, 256, 3) slot for file idx.
                # Numpy's slice assignment copies the rgb array into that slot
                # in place — no reallocation, no overhead.
            except Exception as e:
                # Single-process mode catches per-file errors and continues.
                failed.append((filepath, str(e)))
    else:
        # Multi-process path: parallelize across cores.
        with mp.Pool(processes=num_workers) as pool:
            # pool.imap_unordered:
            #   - Takes a function and an iterable.
            #   - Distributes items across worker processes.
            #   - Yields results as they complete (in any order — hence "unordered").
            #   - This means index N might come back before index N-1, but that's
            #     fine because each result carries its own idx.
            #
            # chunksize=8 means each worker is given 8 files at a time before
            # asking for more. Larger chunksize = less inter-process communication
            # but worse load balancing if files vary in processing time. For our
            # uniform-sized files, 8 is a good default.
            iterator = pool.imap_unordered(_process_one_file, work_items, chunksize=8)

            # tqdm gives us a progress bar.
            # total=n_files tells tqdm how many items to expect (otherwise
            # imap_unordered doesn't advertise its length).
            with tqdm(total=n_files, desc='Processing') as pbar:
                for idx, rgb in iterator:
                    # Place this worker's result into the right slot.
                    # Even though workers finish out of order, the idx field
                    # ensures correct placement.
                    output[idx] = rgb
                    pbar.update(1)
        # Note: with imap_unordered exceptions abort the pool. If finer-grained
        # error handling is needed in future, switch to map_async with callbacks.

    # ─── Save to disk ─────────────────────────────────────────────────────
    # Make sure the parent directory exists. "or '.'" handles the edge case
    # where output_path has no directory part (just a filename in current dir).
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # np.save writes the array in NumPy's native binary format (.npy).
    # Pros: fast load, preserves dtype and shape exactly, no compression
    # overhead. Cons: no compression — file is the full 2 GB on disk.
    # For datasets of this size, fast load > smaller file.
    np.save(output_path, output)

    summary = {
        'n_files': n_files,
        'output_shape': output.shape,
        'output_path': output_path,
        'failed': failed,
    }

    print()
    print(f"Saved: {output_path}")
    print(f"Shape: {output.shape}, size: {output.nbytes / 1e9:.2f} GB")
    if failed:
        print(f"WARNING: {len(failed)} files failed:")
        for fp, err in failed[:10]:
            print(f"  {fp}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    return summary