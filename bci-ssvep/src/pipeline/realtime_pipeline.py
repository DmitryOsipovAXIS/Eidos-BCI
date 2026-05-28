from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from pipeline.pipeline import SSVEPPipeline

class RealtimeCCAPipeline:
    def __init__(self, stream, frequencies_hz: list[float], sample_rate_hz: float =125.0, window_s: float = 2.0, step_s: float = 1.0, n_harmonics: int = 3, labels: list[str]|None = None, confidence_ratio: float = 1.3, min_absolute: float = 0.02,) -> None:
        self._stream = stream
        self._window_samples = int(window_s * sample_rate_hz)
        self._step_s = step_s
        self._labels = labels or ["LEFT", "RIGHT"]
        self._frequencies_hz = frequencies_hz
        self._confidence_ratio = confidence_ratio
        self._min_absolute = min_absolute

        self._pipeline = SSVEPPipeline(
            frequencies_hz=frequencies_hz,
            sample_rate_hz=sample_rate_hz,
            n_harmonics=n_harmonics,
            labels=self._labels
        )

        self._lock = threading.Lock()
        self._latest_label: str = "---"
        self._latest_scores: list[float] = []
        self._latest_peak_hz: float = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None