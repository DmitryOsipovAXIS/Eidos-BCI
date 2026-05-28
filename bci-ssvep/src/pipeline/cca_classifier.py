"""CCA-based SSVEP frequency classifier.

Canonical Correlation Analysis (CCA) finds a linear combination of the multichannel
EEG that maximises correlation with a set of sinusoidal reference signals built at
each stimulus frequency and its harmonics.  The predicted class is the frequency
whose reference achieves the highest canonical correlation with the EEG epoch.

Reference signals Y_f are constructed as:

    Y_f = [ sin(2π f t),  cos(2π f t),
            sin(2π 2f t), cos(2π 2f t),
            ...
            sin(2π Nf t), cos(2π Nf t) ]   shape: (2N, n_samples)

where N is the number of harmonics and t is the time axis.

CCA then solves:

    max_{w_x, w_y}  corr(w_x^T X,  w_y^T Y_f)

The first canonical correlation coefficient ρ_f is used as the detection score.
The stimulus frequency with the highest ρ_f is returned as the prediction.

No training data are needed — CCA is entirely unsupervised at inference time.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import CCA


def build_reference_signals(
    frequency_hz: float,
    n_samples: int,
    sample_rate_hz: float,
    n_harmonics: int = 3,
) -> np.ndarray:
    """Construct sinusoidal reference matrix for one stimulus frequency.

    Args:
        frequency_hz: Fundamental stimulus frequency in Hz.
        n_samples: Number of time points in the analysis window.
        sample_rate_hz: EEG sampling rate in Hz.
        n_harmonics: Number of harmonics to include (including fundamental).

    Returns:
        Array of shape ``(2 * n_harmonics, n_samples)``.
    """
    t = np.arange(n_samples) / sample_rate_hz
    rows = []
    for h in range(1, n_harmonics + 1):
        rows.append(np.sin(2 * np.pi * h * frequency_hz * t))
        rows.append(np.cos(2 * np.pi * h * frequency_hz * t))
    return np.array(rows)


def cca_correlation(
    eeg: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Return the first canonical correlation coefficient between EEG and reference.

    Args:
        eeg: ``(n_channels, n_samples)`` — the EEG epoch.
        reference: ``(2 * n_harmonics, n_samples)`` — sinusoidal reference signals.

    Returns:
        First canonical correlation value in [0, 1].
    """
    X = eeg.T        # (n_samples, n_channels)
    Y = reference.T  # (n_samples, 2 * n_harmonics)

    n_components = 1
    cca = CCA(n_components=n_components, max_iter=1000)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    rho = float(np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1])
    return abs(rho)


class SSVEPClassifierCCA:
    """Classify SSVEP frequency via Canonical Correlation Analysis.

    No training required — pass stimulus frequencies at construction time.

    Args:
        frequencies_hz: Ordered list of stimulus frequencies, e.g. ``[8.0, 15.0]``.
            The index in this list is used as the class label.
        sample_rate_hz: EEG sampling rate in Hz.
        n_harmonics: Number of harmonics (including fundamental) in the reference.
            Rule of thumb: include harmonics up to half the sampling rate.
            At 125 Hz (Nyquist 62.5 Hz):
                8 Hz → 7 harmonics (8,16,24,32,40,48,56 Hz)
                15 Hz → 4 harmonics (15,30,45,60 Hz)
            Using 3 is a safe conservative default for both.
    """

    def __init__(
        self,
        frequencies_hz: list[float],
        sample_rate_hz: float = 125.0,
        n_harmonics: int = 3,
    ) -> None:
        self.frequencies_hz = frequencies_hz
        self.sample_rate_hz = sample_rate_hz
        self.n_harmonics = n_harmonics

    def predict(self, eeg: np.ndarray) -> tuple[int, list[float]]:
        """Classify a single EEG epoch.

        Args:
            eeg: ``(n_channels, n_samples)`` epoch after preprocessing.

        Returns:
            ``(class_index, correlations)`` where ``class_index`` is the index of
            the winning frequency in ``self.frequencies_hz`` and ``correlations``
            is the per-frequency CCA score list.
        """
        n_samples = eeg.shape[1]
        correlations: list[float] = []
        for f in self.frequencies_hz:
            ref = build_reference_signals(f, n_samples, self.sample_rate_hz, self.n_harmonics)
            rho = cca_correlation(eeg, ref)
            correlations.append(rho)
        class_index = int(np.argmax(correlations))
        return class_index, correlations

    def predict_label(self, eeg: np.ndarray, labels: list[str] | None = None) -> tuple[str, list[float]]:
        """Like ``predict`` but returns a string label.

        Args:
            eeg: ``(n_channels, n_samples)`` epoch.
            labels: Human-readable names for each frequency.
                Defaults to ``["LEFT", "RIGHT", ...]``.

        Returns:
            ``(label, correlations)``.
        """
        if labels is None:
            defaults = ["LEFT", "RIGHT", "UP", "DOWN"]
            labels = defaults[: len(self.frequencies_hz)]
        idx, corrs = self.predict(eeg)
        return labels[idx], corrs
