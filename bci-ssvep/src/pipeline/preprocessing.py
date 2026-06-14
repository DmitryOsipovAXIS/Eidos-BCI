"""EEG preprocessing utilities: rereferencing and downsampling.

Rereferencing:
    EEG is recorded relative to a hardware reference electrode.  Common average
    rereferencing (CAR) subtracts the mean across all channels at each time point,
    which attenuates noise that appears identically on every channel (e.g. common-
    mode noise, movement artefacts).  This should be the very first step after
    acquisition — before any filtering — because the filters assume the signal is
    already in a meaningful reference frame.

Downsampling:
    The Neuropawn Knight records at 125 Hz.  Given that the SSVEP bandpass is
    3-30 Hz, 125 Hz is already sufficient (Nyquist = 62.5 Hz >> 30 Hz) and no
    further downsampling is strictly necessary.  The function is provided in case
    a lower rate is ever needed (e.g. to speed up CCA on long windows).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly, detrend
from math import gcd


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """Detrend, then subtract the cross-channel mean at every sample (CAR rereferencing).

    Args:
        data: ``(n_channels, n_samples)`` EEG array.

    Returns:
        Rereferenced array of the same shape.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D (n_channels, n_samples), got shape {data.shape}")
    data = detrend(data, axis=-1)
    return data - data.mean(axis=0, keepdims=True)


def downsample(
    data: np.ndarray,
    original_rate_hz: float,
    target_rate_hz: float,
) -> np.ndarray:
    """Polyphase rational downsampling along the last axis.

    Uses ``scipy.signal.resample_poly`` which applies an anti-aliasing FIR filter
    before decimation, so no extra low-pass step is needed.

    Args:
        data: ``(..., n_samples)`` array.
        original_rate_hz: Current sampling rate in Hz.
        target_rate_hz: Desired sampling rate in Hz. Must be <= original_rate_hz.

    Returns:
        Resampled array; last dimension changes proportionally.
    """
    if target_rate_hz > original_rate_hz:
        raise ValueError("target_rate_hz must be <= original_rate_hz")
    if target_rate_hz == original_rate_hz:
        return data
    up = int(target_rate_hz)
    down = int(original_rate_hz)
    g = gcd(up, down)
    return resample_poly(data, up // g, down // g, axis=-1)
