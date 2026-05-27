"""Offline SSVEP processing pipeline.

Wires together all pipeline stages in the order recommended by the expert:

    raw EEG
      └─► common average rereference   (preprocessing.py)
      └─► notch filter @ 50 Hz         (signal_filtering.py)
      └─► bandpass filter 3-30 Hz      (signal_filtering.py)
      └─► CCA classification           (cca_classifier.py)

Usage (offline / batch):

    pipeline = SSVEPPipeline(frequencies_hz=[8.0, 15.0], sample_rate_hz=125.0)

    # Single epoch: (n_channels, n_samples)
    label, scores = pipeline.run(epoch)

    # Batch of epochs: (n_trials, n_channels, n_samples)
    labels, scores_list = pipeline.run_batch(epochs)
"""

from __future__ import annotations

import numpy as np

from pipeline.cca_classifier import SSVEPClassifierCCA
from pipeline.preprocessing import common_average_reference
from pipeline.signal_filtering import apply_filters


class SSVEPPipeline:
    """End-to-end offline SSVEP pipeline (no training required).

    Args:
        frequencies_hz: Stimulus frequencies in the order that defines class labels.
            Index 0 → LEFT, index 1 → RIGHT by convention.
        sample_rate_hz: EEG sampling rate in Hz.
        n_harmonics: Harmonics passed to CCA reference construction.
        notch_hz: AC line frequency to suppress (50 Hz Europe).
        bandpass_low_hz: Lower edge of bandpass filter (Hz).
        bandpass_high_hz: Upper edge of bandpass filter (Hz).
        labels: Optional human-readable class names matching ``frequencies_hz``.
    """

    def __init__(
        self,
        frequencies_hz: list[float] = None,
        sample_rate_hz: float = 125.0,
        n_harmonics: int = 3,
        notch_hz: float = 50.0,
        bandpass_low_hz: float = 3.0,
        bandpass_high_hz: float = 40.0,
        labels: list[str] | None = None,
    ) -> None:
        if frequencies_hz is None:
            frequencies_hz = [8.0, 15.0]
        self.frequencies_hz = frequencies_hz
        self.sample_rate_hz = sample_rate_hz
        self.notch_hz = notch_hz
        self.bandpass_low_hz = bandpass_low_hz
        self.bandpass_high_hz = bandpass_high_hz
        self.labels = labels or ["LEFT", "RIGHT", "UP", "DOWN"][: len(frequencies_hz)]
        self.classifier = SSVEPClassifierCCA(
            frequencies_hz=frequencies_hz,
            sample_rate_hz=sample_rate_hz,
            n_harmonics=n_harmonics,
        )

    def preprocess(self, eeg: np.ndarray) -> np.ndarray:
        """Apply CAR rereferencing and sequential filtering to one epoch.

        Args:
            eeg: ``(n_channels, n_samples)``

        Returns:
            Preprocessed array of the same shape.
        """
        eeg = common_average_reference(eeg)
        eeg = apply_filters(
            eeg,
            sample_rate_hz=self.sample_rate_hz,
            notch_hz=self.notch_hz,
            bandpass_low_hz=self.bandpass_low_hz,
            bandpass_high_hz=self.bandpass_high_hz,
        )
        return eeg

    def run(self, eeg: np.ndarray) -> tuple[str, list[float]]:
        """Process one epoch and return the predicted label and CCA scores.

        Args:
            eeg: Raw EEG epoch ``(n_channels, n_samples)``.

        Returns:
            ``(label, cca_scores)`` — e.g. ``("LEFT", [0.82, 0.41])``.
        """
        processed = self.preprocess(eeg)
        return self.classifier.predict_label(processed, self.labels)

    def run_batch(
        self, epochs: np.ndarray
    ) -> tuple[list[str], list[list[float]]]:
        """Process a batch of epochs.

        Args:
            epochs: ``(n_trials, n_channels, n_samples)``

        Returns:
            ``(labels, cca_scores_per_trial)``
        """
        labels: list[str] = []
        scores: list[list[float]] = []
        for epoch in epochs:
            lbl, sc = self.run(epoch)
            labels.append(lbl)
            scores.append(sc)
        return labels, scores

    def evaluate(
        self,
        epochs: np.ndarray,
        true_labels: list[str] | np.ndarray,
    ) -> dict:
        """Run batch prediction and compute accuracy.

        Args:
            epochs: ``(n_trials, n_channels, n_samples)``
            true_labels: Ground-truth label per trial (strings matching ``self.labels``).

        Returns:
            Dict with keys ``accuracy``, ``n_correct``, ``n_total``, ``predictions``.
        """
        pred_labels, scores = self.run_batch(epochs)
        n_total = len(true_labels)
        n_correct = sum(p == t for p, t in zip(pred_labels, true_labels))
        return {
            "accuracy": n_correct / n_total if n_total > 0 else 0.0,
            "n_correct": n_correct,
            "n_total": n_total,
            "predictions": pred_labels,
            "cca_scores": scores,
        }
