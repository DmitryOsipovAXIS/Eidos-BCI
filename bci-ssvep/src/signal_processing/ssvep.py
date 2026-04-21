"""Windowing and small helpers for SSVEP epoch handling."""

from __future__ import annotations

import numpy as np


def slice_windows(
    eeg: np.ndarray,
    window_samples: int,
    step_samples: int,
) -> np.ndarray:
    """Cut a continuous EEG segment into overlapping windows.

    Args:
        eeg: Shape ``(n_channels, n_samples)``.
        window_samples: Window length in samples.
        step_samples: Step between consecutive window starts.

    Returns:
        Array of shape ``(n_windows, n_channels, window_samples)``.
    """
    n_ch, n_s = eeg.shape
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("window_samples and step_samples must be positive")
    if n_s < window_samples:
        return np.empty((0, n_ch, window_samples), dtype=eeg.dtype)

    starts = np.arange(0, n_s - window_samples + 1, step_samples, dtype=int)
    out = np.stack([eeg[:, s : s + window_samples] for s in starts], axis=0)
    return out
