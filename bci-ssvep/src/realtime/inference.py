"""Real-time SSVEP decoding: sliding window, feature extraction, sklearn inference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
from brainflow.board_shim import BoardIds

from acquisition.brainflow_stream import BrainFlowStream
from features.frequency_features import extract_ssvep_feature_vector
from models.classifier import load_sklearn_pipeline
from signal_processing.filters import bandpass_filter
from utils.config import SSVEPConfig


def predict_side_label(class_index: int) -> str:
    """Map sklearn class 0/1 to a human-readable motor metaphor."""
    return "LEFT" if int(class_index) == 0 else "RIGHT"


class RealtimeSSVEPDecoder:
    """Load a trained pipeline and decode SSVEP from a live BrainFlow stream."""

    def __init__(
        self,
        model_path: str | Path,
        config: Optional[SSVEPConfig] = None,
        board_id: Optional[int] = None,
    ) -> None:
        self.config = config or SSVEPConfig()
        self.model: Any = load_sklearn_pipeline(model_path)
        self.board_id = int(board_id) if board_id is not None else int(BoardIds.SYNTHETIC_BOARD)

    def _preprocess(self, eeg: np.ndarray, fs: float) -> np.ndarray:
        low, high = self.config.bandpass_band_hz()
        return bandpass_filter(eeg, fs, low, high)

    def featurize_window(self, eeg: np.ndarray, fs: float) -> np.ndarray:
        """Return 1D feature vector for one window ``(n_channels, n_samples)``."""
        proc = self._preprocess(eeg, fs)
        return extract_ssvep_feature_vector(
            proc,
            fs,
            (self.config.left_hz, self.config.right_hz),
        )

    def predict_window(self, eeg: np.ndarray, fs: float) -> tuple[str, int]:
        x = self.featurize_window(eeg, fs).reshape(1, -1)
        cls = int(self.model.predict(x)[0])
        return predict_side_label(cls), cls

    def run_forever(
        self,
        poll_interval_s: float = 0.05,
    ) -> Iterator[tuple[str, int]]:
        """Yield ``(side, class_index)`` for successive analysis windows.

        Each prediction consumes buffered data and waits for a fresh window
        of length ``config.window_seconds`` to accumulate again.
        """
        stream = BrainFlowStream(board_id=self.board_id)
        stream.prepare_session()
        stream.start_stream()
        fs = stream.sampling_rate()
        window_samples = int(self.config.window_seconds * fs)
        refill_sleep = max(self.config.window_seconds * 0.98, 0.1)
        try:
            while True:
                if stream.get_board_data_count() < window_samples:
                    time.sleep(poll_interval_s)
                    continue
                raw = stream.get_current_board_data(window_samples)
                eeg_idx = stream.eeg_channel_indices()
                eeg = raw[eeg_idx, :]
                side, cls = self.predict_window(eeg, fs)
                yield side, cls
                stream.get_board_data()
                time.sleep(refill_sleep)
        finally:
            stream.stop_stream()
            stream.release_session()
