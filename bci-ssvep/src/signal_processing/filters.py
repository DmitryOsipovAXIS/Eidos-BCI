"""Simple bandpass filtering for SSVEP band (Butterworth + zero-phase)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    data: np.ndarray,
    sample_rate_hz: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass along the last axis (time).

    Args:
        data: Shape ``(..., n_samples)``; filtering is applied independently on the last axis.
        sample_rate_hz: Sampling frequency in Hz.
        low_hz: Low cutoff (Hz).
        high_hz: High cutoff (Hz).
        order: Butterworth filter order.

    Returns:
        Filtered array with the same shape as ``data``.
    """
    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError("Require 0 < low_hz < high_hz")
    nyq = 0.5 * sample_rate_hz
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)
    if low >= high:
        raise ValueError("Invalid band after Nyquist normalization")

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=-1)
