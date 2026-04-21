"""BrainFlow board lifecycle and helpers for streaming / labeled collection."""

from __future__ import annotations

import time
from typing import Iterable, Optional

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

from utils.config import SSVEPConfig


class BrainFlowStream:
    """Thin wrapper around ``BoardShim`` for EEG streaming.

    By default uses ``BoardIds.SYNTHETIC_BOARD`` so the pipeline runs without hardware.
    """

    def __init__(
        self,
        board_id: Optional[int] = None,
        params: Optional[BrainFlowInputParams] = None,
        log_board: bool = False,
    ) -> None:
        self.board_id = int(board_id) if board_id is not None else int(BoardIds.SYNTHETIC_BOARD)
        self.params = params or BrainFlowInputParams()
        self._board: Optional[BoardShim] = None

        if log_board:
            BoardShim.enable_board_logger()

    @property
    def board(self) -> BoardShim:
        if self._board is None:
            raise RuntimeError("Session not prepared. Call prepare_session() first.")
        return self._board

    def sampling_rate(self) -> float:
        return float(BoardShim.get_sampling_rate(self.board_id))

    def eeg_channel_indices(self) -> list[int]:
        return list(BoardShim.get_eeg_channels(self.board_id))

    def prepare_session(self) -> None:
        self._board = BoardShim(self.board_id, self.params)
        self._board.prepare_session()

    def release_session(self) -> None:
        if self._board is not None:
            self._board.release_session()
            self._board = None

    def start_stream(self, buffer_size_samples: int = 450000) -> None:
        self.board.start_stream(buffer_size_samples)

    def stop_stream(self) -> None:
        self.board.stop_stream()

    def get_board_data(self) -> np.ndarray:
        """Return all samples currently in the buffer (channels x samples) and clear it."""
        return self.board.get_board_data()

    def get_current_board_data(self, num_samples: int) -> np.ndarray:
        """Peek at the last ``num_samples`` without consuming the whole buffer."""
        return self.board.get_current_board_data(num_samples)

    def get_board_data_count(self) -> int:
        return self.board.get_board_data_count()

    def __enter__(self) -> "BrainFlowStream":
        self.prepare_session()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.stop_stream()
        except Exception:
            pass
        self.release_session()


def collect_labeled_window(
    stream: BrainFlowStream,
    label: int,
    duration_s: float,
    discard_initial_s: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Record one contiguous EEG window with an integer label.

    Args:
        stream: Prepared and streaming ``BrainFlowStream`` (``start_stream`` already called).
        label: 0 = LEFT (``config.left_hz``), 1 = RIGHT (``config.right_hz``) in downstream mapping.
        duration_s: Wall-clock length of the recording window.
        discard_initial_s: Drop the first part of the window (onset transients).

    Returns:
        Tuple ``(eeg, label)`` where ``eeg`` has shape ``(n_eeg_channels, n_samples)``.
    """
    if duration_s <= discard_initial_s:
        raise ValueError("duration_s must be greater than discard_initial_s")

    fs = stream.sampling_rate()
    stream.get_board_data()  # flush buffer

    time.sleep(duration_s)
    raw = stream.get_board_data()
    eeg_idx = stream.eeg_channel_indices()
    eeg = raw[eeg_idx, :]

    start = int(discard_initial_s * fs)
    if eeg.shape[1] <= start:
        raise RuntimeError(
            f"Insufficient samples after discard: got {eeg.shape[1]}, "
            f"need more than {start} (check duration / sampling rate)."
        )
    eeg = eeg[:, start:]
    return eeg, label


def collect_labeled_windows(
    stream: BrainFlowStream,
    labels: Iterable[int],
    duration_s: float,
    discard_initial_s: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Record one window per label value (sequential, same stream session).

    Returns:
        ``X`` with shape ``(n_trials, n_channels, n_samples)`` and ``y`` of shape ``(n_trials,)``.
    """
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for lab in labels:
        seg, _ = collect_labeled_window(stream, int(lab), duration_s, discard_initial_s)
        xs.append(seg)
        ys.append(int(lab))

    min_len = min(x.shape[1] for x in xs)
    trimmed = [x[:, :min_len] for x in xs]
    X = np.stack(trimmed, axis=0)
    y = np.array(ys, dtype=np.int64)
    return X, y


def collect_labeled_data_cli(
    stream: BrainFlowStream,
    config: SSVEPConfig,
    seconds_per_class: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Interactive CLI: collect one window per class (LEFT then RIGHT).

    Returns:
        ``X`` with shape ``(n_trials, n_channels, n_samples)`` and ``y`` shape ``(n_trials,)``.
    """
    stream.prepare_session()
    stream.start_stream()
    try:
        input(
            f"\nFocus LEFT target ({config.left_hz} Hz). Press Enter to record "
            f"{seconds_per_class:.1f}s...\n"
        )
        left_eeg, _ = collect_labeled_window(stream, 0, seconds_per_class)

        input(
            f"\nFocus RIGHT target ({config.right_hz} Hz). Press Enter to record "
            f"{seconds_per_class:.1f}s...\n"
        )
        right_eeg, _ = collect_labeled_window(stream, 1, seconds_per_class)
    finally:
        stream.stop_stream()
        stream.release_session()

    min_len = min(left_eeg.shape[1], right_eeg.shape[1])
    X = np.stack([left_eeg[:, :min_len], right_eeg[:, :min_len]], axis=0)
    y = np.array([0, 1], dtype=np.int64)
    return X, y
