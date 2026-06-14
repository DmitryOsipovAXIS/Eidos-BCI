"""EEG board initialisation and recording helpers."""
from __future__ import annotations

import dataclasses
import threading
import time
from typing import Optional

import numpy as np


REC_THREAD_START_TIMEOUT_S = 2


@dataclasses.dataclass
class RecordingResult:
    eeg: Optional[np.ndarray] = None
    label: Optional[int] = None
    error: Optional[str] = None


def _record_window(
    stream,
    label: int,
    duration_s: float,
    discard_s: float,
    start_event: threading.Event,
    result: RecordingResult,
) -> None:
    from acquisition.brainflow_stream import collect_labeled_window
    try:
        start_event.wait(timeout=duration_s + REC_THREAD_START_TIMEOUT_S)
        eeg, lbl = collect_labeled_window(stream, label, duration_s, discard_s)
        result.eeg = eeg
        result.label = lbl
    except Exception as e:
        result.error = str(e)


def start_recording_thread(
    stream,
    label: int,
    duration_s: float,
    discard_s: float,
) -> tuple[threading.Thread, threading.Event, RecordingResult]:
    result = RecordingResult()
    start_event = threading.Event()
    thread = threading.Thread(
        target=_record_window,
        args=(stream, label, duration_s, discard_s, start_event, result),
        daemon=True,
    )
    thread.start()
    return thread, start_event, result


def init_board(args) -> tuple[object, float]:
    from acquisition.brainflow_stream import BrainFlowStream
    from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

    BoardShim.disable_board_logger()
    p = BrainFlowInputParams()
    if args.serial_port:
        p.serial_port = args.serial_port

    stream = BrainFlowStream(
        board_id=BoardIds.NEUROPAWN_KNIGHT_BOARD.value,
        params=p,
        num_channels=args.num_channels,
    )
    stream.prepare_session()
    stream.start_stream()
    time.sleep(2)

    for ch in range(1, args.num_channels + 1):
        time.sleep(0.25)
        try:
            stream.board.config_board(f"chon_{ch}_12")
        except Exception:
            pass
        time.sleep(0.25)
        try:
            stream.board.config_board(f"rldadd_{ch}")
        except Exception:
            pass
        time.sleep(0.25)

    fs = stream.sampling_rate()
    if abs(fs - 125.0) > 1e-6:
        stream.stop_stream()
        stream.release_session()
        raise RuntimeError(
            f"Board sampling rate is {fs:0.3f} Hz, expected 125.000 Hz")

    print(f"Board ready at {fs:0.3f} Hz", flush=True)
    return stream, fs
