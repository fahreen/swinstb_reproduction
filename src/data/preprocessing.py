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
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

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
    idx, filepath = args
    iq = parse_iq_file(filepath)
    rgb = iq_to_spectrogram(iq)
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
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            f"Pass overwrite=True to replace it."
        )

    files = list_files_chronological(input_dir)
    if not files:
        raise ValueError(f"No .xls files found in {input_dir}")

    n_files = len(files)
    print(f"Found {n_files} files in {input_dir}")
    print(f"Output: {output_path}")

    # Allocate output array
    output = np.zeros((n_files, output_size, output_size, 3), dtype=np.uint8)
    print(f"Allocated output array: shape {output.shape} "
          f"(~{output.nbytes / 1e9:.2f} GB)")

    # Decide worker count
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Using {num_workers} worker process{'es' if num_workers != 1 else ''}")
    print()

    # Build work items: (index, filepath) tuples
    work_items = list(enumerate(files))

    # Track failures so we can report them at the end
    failed: List[Tuple[str, str]] = []

    # Run the pool with a progress bar
    if num_workers == 1:
        # Single-process path — easier to debug than launching a pool of 1
        for idx, filepath in tqdm(work_items, total=n_files, desc='Processing'):
            try:
                _, rgb = _process_one_file((idx, filepath))
                output[idx] = rgb
            except Exception as e:
                failed.append((filepath, str(e)))
    else:
        # Multi-process path
        with mp.Pool(processes=num_workers) as pool:
            iterator = pool.imap_unordered(_process_one_file, work_items, chunksize=8)
            with tqdm(total=n_files, desc='Processing') as pbar:
                for idx, rgb in iterator:
                    output[idx] = rgb
                    pbar.update(1)
        # Note: with imap_unordered exceptions abort the pool. If finer-grained
        # error handling is needed in future, switch to map_async with callbacks.

    # Ensure the output directory exists, then save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
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