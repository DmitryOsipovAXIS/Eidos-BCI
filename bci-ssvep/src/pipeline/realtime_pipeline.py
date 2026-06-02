from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import Counter
from typing import Optional

import numpy as np

from pipeline.pipeline import SSVEPPipeline

class RealtimeCCAPipeline:
    def __init__(self, stream, frequencies_hz: list[float], sample_rate_hz: float =125.0, window_s: float = 4.0, step_s: float = 0.5, n_harmonics: int = 3, labels: list[str]|None = None, confidence_ratio: float = 1.3, min_absolute: float = 0.02, ws_loop: Optional[asyncio.AbstractEventLoop] = None, ws_broadcast=None,) -> None:
        self._stream = stream
        self._window_samples = int(window_s * sample_rate_hz)
        self._step_s = step_s
        self._labels = labels or ["LEFT", "RIGHT"]
        self._frequencies_hz = frequencies_hz
        self._confidence_ratio = confidence_ratio
        self._min_absolute = min_absolute
        self._ws_loop = ws_loop
        self._ws_broadcast = ws_broadcast
        self._last_sent: str = ""
        self._history: list[str] = []
        self._all_guesses: list[str] = []
        self._vote_window: int = 5
        self._vote_threshold: int = 3

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

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
    
    def get_latest(self) -> tuple[str, list[float], float]:
        with self._lock:
            return self._latest_label, list(self._latest_scores), self._latest_peak_hz

    def get_full_history(self) -> list[str]:
        with self._lock:
            return list(self._all_guesses)
        
    def is_confident(self, scores:list[float]) -> bool:
        arr = np.array(scores)
        if len(arr) < 2:
            return False
        best = float(np.max(arr))
        if best < self._min_absolute:
            return False
        second = float(np.partition(arr, -2)[-2])
        return (best / (second + 1e-9)) >= self._confidence_ratio
    
    def _loop(self) -> None:
        while self._running:
            try:
                if self._stream.get_board_data_count() < self._window_samples:
                    time.sleep(0.05)
                    continue

                raw = self._stream.get_current_board_data(self._window_samples)
                eeg_idx = self._stream.eeg_channel_indices()
                eeg = raw[eeg_idx, :]

                if eeg.shape[1] < self._window_samples:
                    time.sleep(0.05)
                    continue

                label, scores = self._pipeline.run(eeg)
                if not self.is_confident(scores):
                    label = "NEUTRAL"
                peak_hz = self._frequencies_hz[int(np.argmax(scores))]

                self._history.append(label)
                self._all_guesses.append(label)
                if len(self._history) > self._vote_window:
                    self._history.pop(0)

                counts = Counter(self._history)
                stable_label = counts.most_common(1)[0][0]
                stable_enough = counts[stable_label] >= self._vote_threshold

                with self._lock:
                    self._latest_label = stable_label if stable_enough else "---"
                    self._latest_scores = scores
                    self._latest_peak_hz = peak_hz

                if stable_enough and stable_label != self._last_sent and self._ws_loop is not None and self._ws_broadcast is not None:
                    event_type = "FUNCTION" if stable_label == "LEFT" else "WIDGET"
                    msg = json.dumps({"type": event_type, "direction": stable_label, "peak_hz": peak_hz})
                    asyncio.run_coroutine_threadsafe(self._ws_broadcast(msg), self._ws_loop)
                    self._last_sent = stable_label

            except Exception:
                pass

            time.sleep(self._step_s)