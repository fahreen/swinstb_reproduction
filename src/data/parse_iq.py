"""
Parse raw .xls files from the NUAA FM/LTE dataset.

Despite the .xls extension, these are plain-text files with tab-separated values
and Windows-style line endings (CRLF).

File structure:
    Row 0 (header): 6 tab-separated fields
        [0] I0       — first I sample (signed integer, 14-bit)
        [1] Q0       — first Q sample (signed integer, 14-bit)
        [2] HHMMSS   — recording timestamp, matching the filename suffix
        [3] longitude (east, decimal degrees)
        [4] latitude  (north, decimal degrees)
        [5] altitude  (metres)

    Rows 1 to N-1: I Q pairs, one per row, tab-separated signed integers.

Total samples per file:
    FM:  32,508 (header counts as sample 0; rows 1-32507 are samples 1-32507)
    LTE: 16,254

Note: the paper says "STFT number = 32,508" — this matches the total line count
exactly when the header's I0/Q0 are counted as the first sample.
"""

import os
import re
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FM_SAMPLES_PER_FILE = 32508
LTE_SAMPLES_PER_FILE = 16254

# 14-bit signed ADC range
ADC_MIN = -8192
ADC_MAX = 8191

# Filename pattern: 172.19.220.14-2022-0923-HHMMSS.xls
# Example: 172.19.220.14-2022-0923-092023.xls
_FILENAME_TIMESTAMP_RE = re.compile(r'.*-(\d{6})\.xls$')


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_iq_file(filepath: str, expected_samples: Optional[int] = None) -> np.ndarray:
    """
    Parse one .xls file and return its complex I/Q samples.

    Args:
        filepath: Path to the .xls file.
        expected_samples: If given, raise ValueError if the file's sample count
            doesn't match. Pass FM_SAMPLES_PER_FILE or LTE_SAMPLES_PER_FILE
            for strict validation.

    Returns:
        Complex array of shape (N,), dtype complex128, where N is the total
        sample count (32508 for FM, 16254 for LTE). The first sample is
        constructed from the header's I0/Q0 fields; subsequent samples come
        from rows 1 onward.

    Raises:
        ValueError: if file format is malformed or sample count mismatches
            expected_samples.
    """
    with open(filepath, 'r') as f:
        # rstrip removes both the trailing \n and any \r before it.
        # This handles Windows CRLF line endings cleanly.
        lines = [line.rstrip() for line in f if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"File {filepath} has fewer than 2 lines")

    # Parse header
    header_fields = lines[0].split('\t')
    if len(header_fields) != 6:
        raise ValueError(
            f"File {filepath}: header has {len(header_fields)} fields, expected 6. "
            f"Got: {header_fields}"
        )

    try:
        i0 = int(header_fields[0])
        q0 = int(header_fields[1])
    except ValueError as e:
        raise ValueError(
            f"File {filepath}: could not parse I0/Q0 from header: {header_fields[:2]}"
        ) from e

    # Parse body rows
    n_body = len(lines) - 1
    i_samples = np.empty(1 + n_body, dtype=np.int32)
    q_samples = np.empty(1 + n_body, dtype=np.int32)
    i_samples[0] = i0
    q_samples[0] = q0

    for idx, line in enumerate(lines[1:], start=1):
        parts = line.split('\t')
        if len(parts) != 2:
            raise ValueError(
                f"File {filepath}, row {idx}: expected 2 tab-separated values, "
                f"got {len(parts)}: {parts}"
            )
        try:
            i_samples[idx] = int(parts[0])
            q_samples[idx] = int(parts[1])
        except ValueError as e:
            raise ValueError(
                f"File {filepath}, row {idx}: could not parse I/Q values: {parts}"
            ) from e

    total = 1 + n_body
    if expected_samples is not None and total != expected_samples:
        raise ValueError(
            f"File {filepath}: got {total} samples, expected {expected_samples}"
        )

    # Build complex baseband signal
    return i_samples.astype(np.float64) + 1j * q_samples.astype(np.float64)


def get_file_timestamp(filepath: str) -> str:
    """
    Extract the HHMMSS timestamp from a NUAA filename.

    Args:
        filepath: Path or filename like '172.19.220.14-2022-0923-092023.xls'.

    Returns:
        Six-character string like '092023'. Useful as a chronological sort key.

    Raises:
        ValueError: if the filename doesn't match the expected pattern.
    """
    name = os.path.basename(filepath)
    match = _FILENAME_TIMESTAMP_RE.match(name)
    if match is None:
        raise ValueError(
            f"Filename does not match expected pattern '...HHMMSS.xls': {name}"
        )
    return match.group(1)


def list_files_chronological(directory: str) -> list:
    """
    List all .xls files in a directory, sorted chronologically by filename.

    Files whose names don't match the expected pattern are silently skipped.

    Args:
        directory: Path to a directory containing NUAA .xls files.

    Returns:
        List of full paths, sorted by HHMMSS timestamp ascending.
    """
    paths = []
    for entry in os.listdir(directory):
        if not entry.endswith('.xls'):
            continue
        try:
            get_file_timestamp(entry)  # validates pattern
        except ValueError:
            continue
        paths.append(os.path.join(directory, entry))

    paths.sort(key=get_file_timestamp)
    return paths