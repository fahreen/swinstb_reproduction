"""
 read the raw radio recordings off disk and turn each one into a numpy array of complex numbers


 After calling parse_iq_file(some_xls_path), you get back a 1D numpy array like this:

 
 [ 142+87j, 156+92j, 134+78j, ..., -89-65j ]   # shape (32508,), dtype complex128




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
# NumPy is the foundation of all scientific Python.
# Think of it as: arrays + math operations that run in C-speed under the hood.
# Every PyTorch tensor can convert to a NumPy array, and vice versa.


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Number of I/Q sample pairs per file in each dataset.
# This is a HARDWARE-LEVEL fact: the SDR (Software-Defined Radio) used by NUAA
# was configured to capture exactly this many samples per recording.
# Roughly 1ms of signal time, sampled at 31.25 MHz (125 MHz original rate, decimated by 4).
FM_SAMPLES_PER_FILE = 32508
LTE_SAMPLES_PER_FILE = 16254

# 14-bit signed ADC range.
# An ADC (Analog-to-Digital Converter) is the chip inside the SDR that turns
# the analog radio voltage into integers. "14-bit signed" means each I or Q
# value is an integer in the range [-2^13, 2^13 - 1] = [-8192, 8191].
# This range is fixed by the hardware; the actual signal magnitudes we see
# may fill or be much smaller than this range depending on signal strength.
ADC_MIN = -8192
ADC_MAX = 8191

# Filename pattern: 172.19.220.14-2022-0923-HHMMSS.xls
# Example: 172.19.220.14-2022-0923-092023.xls
#
# The "172.19.220.14" is the SDR's internal IP address (just metadata).
# "2022-0923" is the date (Sept 23, 2022) — all files are from this one day.
# "HHMMSS" is the recording timestamp; this is what we use to sort the files
# chronologically. 092023 means 09:20:23 AM, for example.
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

    # ─── Parse header (first line) ─────────────────────────────────────────
    # The header row contains the very first I/Q sample plus GPS metadata.
    # This is a peculiarity of the NUAA data format: they decided to put
    # sample[0] in the header instead of just the first data row.
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

    # ─── Parse body rows (remaining lines) ─────────────────────────────────
    # Each subsequent line is one I/Q pair.
    n_body = len(lines) - 1

    # np.empty allocates an uninitialized array of the given size.
    # This is FASTER than np.zeros because it skips the zero-fill step.
    # Safe here because we immediately write into every slot below.
    # dtype=np.int32 means 32-bit signed integers (enough headroom for 14-bit values).
    i_samples = np.empty(1 + n_body, dtype=np.int32)
    q_samples = np.empty(1 + n_body, dtype=np.int32)

    # Plant the header's first sample at index 0.
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

    # ─── Build complex baseband signal ─────────────────────────────────────
    #
    # This is the KEY operation of this whole file. Some background:
    #
    # A radio wave is a sinusoid that varies in both AMPLITUDE (how strong)
    # and PHASE (where it is in its cycle). To represent this mathematically,
    # you need TWO numbers per time instant. The radio engineering convention
    # is to call them I (In-phase) and Q (Quadrature), which are 90° apart.
    #
    # The clever trick: pack (I, Q) into a single complex number Z = I + j*Q,
    # where j is the imaginary unit (math people use 'i', engineers use 'j'
    # to avoid clash with current). Once you do this:
    #
    #   - The MAGNITUDE |Z| = sqrt(I² + Q²) = the signal's instantaneous amplitude.
    #   - The PHASE angle(Z) = atan2(Q, I) = the signal's instantaneous phase.
    #
    # This isn't just notation — complex multiplication does meaningful things:
    # multiplying by exp(j*omega*t) shifts the signal in frequency. This is the
    # mathematical basis of the FFT (Fast Fourier Transform), which we'll use
    # next in stft.py.
    #
    # Why .astype(np.float64) first? Because numpy's complex multiplication
    # operates on floats internally. Going int32 → complex128 directly would
    # work but is less explicit. We promote to float64, which has the same
    # bit-width as each half of complex128 (64 + 64 = 128 bits).
    #
    # The returned array has dtype complex128 automatically — numpy sees that
    # one side is real, the other is imaginary, and infers the right dtype.
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
    # Regex match — captures the 6 digits before .xls into group 1.
    match = _FILENAME_TIMESTAMP_RE.match(name)
    if match is None:
        raise ValueError(
            f"Filename does not match expected pattern '...HHMMSS.xls': {name}"
        )
    # Return the timestamp as a string ('092023'), not an int.
    # Strings sort lexicographically and HHMMSS sorts correctly that way
    # (because all parts are zero-padded to fixed width).
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
    # CRITICAL: chronological sorting is essential for forecasting.
    # If files were processed in arbitrary order (e.g., the filesystem's
    # default order, which depends on the OS), then the train/val/test
    # split would be temporally jumbled, and the model could effectively
    # "cheat" by interpolating between adjacent frames in time.
    # By sorting by HHMMSS first, the downstream code can do a clean
    # 4:1:1 chronological split (first 67% = train, then 17% val, then 17% test).
    paths = []
    for entry in os.listdir(directory):
        if not entry.endswith('.xls'):
            continue
        try:
            get_file_timestamp(entry)  # validates pattern
        except ValueError:
            continue
        paths.append(os.path.join(directory, entry))

    # The 'key' argument tells sort() to compare files by their HHMMSS
    # timestamp rather than the full filename. Since all files share the
    # same date and IP prefix, sorting by HHMMSS is equivalent to sorting
    # by full filename in this case, but more explicit about intent.
    paths.sort(key=get_file_timestamp)
    return paths