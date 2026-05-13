"""SSVEP 4-box flicker stimulus with real-time BrainFlow frequency detection.

Four boxes flicker at distinct SSVEP frequencies. When the detected dominant
frequency matches a box's target (within tolerance), that box turns its
highlight color instead of white/black.

Frequencies (all exact divisors of 60 Hz):
  Box 0 (top-left)     :  6 Hz  -> red    (60÷10)
  Box 1 (top-right)    : 10 Hz  -> green  (60÷6)
  Box 2 (bottom-left)  : 12 Hz  -> blue   (60÷5)
  Box 3 (bottom-right) : 15 Hz  -> yellow (60÷4)

Run with --board-id to select a real board (default: synthetic).
Run with --no-eeg to show only the visual stimulus without BrainFlow.
Press F11 or Escape to toggle/exit fullscreen.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tkinter as tk

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOXES = [
    {"freq": 6.0,  "label": "6 Hz",  "color": "#FF4444"},  # red    (60÷10)
    {"freq": 10.0, "label": "10 Hz", "color": "#44FF44"},  # green  (60÷6)
    {"freq": 12.0, "label": "12 Hz", "color": "#4488FF"},  # blue   (60÷5)
    {"freq": 15.0, "label": "15 Hz", "color": "#FFFF44"},  # yellow (60÷4)
]

DETECT_TOLERANCE_HZ = 0.8   # how close detected peak must be to a target freq
WINDOW_SECONDS = 2.0        # EEG analysis window
ANALYSIS_INTERVAL_S = 0.5   # how often to run FFT detection


# ---------------------------------------------------------------------------
# Flicker state
# ---------------------------------------------------------------------------

class FlickerState:
    def __init__(self) -> None:
        self.detected_freq: Optional[float] = None
        self.lock = threading.Lock()

    def set_detected(self, freq: Optional[float]) -> None:
        with self.lock:
            self.detected_freq = freq

    def get_detected(self) -> Optional[float]:
        with self.lock:
            return self.detected_freq


# ---------------------------------------------------------------------------
# EEG analysis thread
# ---------------------------------------------------------------------------

def eeg_analysis_thread(state: FlickerState, board_id: int, stop_event: threading.Event) -> None:
    """Runs in background: streams EEG, computes Welch PSD, updates detected freq."""
    from scipy.signal import welch as scipy_welch

    params = BrainFlowInputParams()
    board = BoardShim(board_id, params)
    BoardShim.disable_board_logger()

    try:
        board.prepare_session()
        board.start_stream(45000)
        fs = float(BoardShim.get_sampling_rate(board_id))
        eeg_channels = list(BoardShim.get_eeg_channels(board_id))
        window_samples = int(WINDOW_SECONDS * fs)

        while not stop_event.is_set():
            time.sleep(ANALYSIS_INTERVAL_S)
            if board.get_board_data_count() < window_samples:
                continue

            raw = board.get_current_board_data(window_samples)
            eeg = raw[eeg_channels, :]  # (n_ch, n_samples)

            # Average Welch PSD across channels
            nperseg = min(256, eeg.shape[1])
            psds = []
            freqs = None
            for ch in range(eeg.shape[0]):
                f, pxx = scipy_welch(eeg[ch], fs=fs, nperseg=nperseg)
                psds.append(pxx)
                freqs = f
            if freqs is None:
                continue

            psd_mean = np.mean(np.stack(psds, axis=0), axis=0)

            # Only look in SSVEP range (4-30 Hz)
            mask = (freqs >= 4.0) & (freqs <= 30.0)
            if not np.any(mask):
                continue

            peak_freq = float(freqs[mask][np.argmax(psd_mean[mask])])

            # Check if peak matches any target
            matched = None
            for box in BOXES:
                if abs(peak_freq - box["freq"]) <= DETECT_TOLERANCE_HZ:
                    matched = box["freq"]
                    break

            state.set_detected(matched)

    finally:
        try:
            board.stop_stream()
        except Exception:
            pass
        try:
            board.release_session()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

# Boxes occupy this fraction of the screen dimension on each axis.
# Leaves a large dark gap between boxes to prevent visual interference.
BOX_FRACTION = 0.30   # each box is 30% of screen width/height
GAP_FRACTION  = 0.10  # gap between boxes is 10% of screen dimension


class SSVEPFlickerApp:

    def __init__(self, root: tk.Tk, state: FlickerState) -> None:
        self.root = root
        self.state = state

        root.title("SSVEP 4-Box Flicker")
        root.configure(bg="#000000")
        root.attributes("-fullscreen", True)

        # F11 toggles fullscreen; Escape exits fullscreen (back to window)
        root.bind("<F11>", self._toggle_fullscreen)
        root.bind("<Escape>", self._exit_fullscreen)

        self.canvas = tk.Canvas(root, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._phases = [0.0] * len(BOXES)
        self._last_time = time.perf_counter()
        self._rects: list[int] = []
        self._labels: list[int] = []
        self._fullscreen = True

        # Build after geometry is resolved
        root.update_idletasks()
        self._build_boxes()
        root.bind("<Configure>", self._on_resize)

        self._tick()

    # ------------------------------------------------------------------
    # Fullscreen helpers
    # ------------------------------------------------------------------

    def _toggle_fullscreen(self, *_) -> None:
        self._fullscreen = not self._fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)

    def _exit_fullscreen(self, *_) -> None:
        self._fullscreen = False
        self.root.attributes("-fullscreen", False)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _on_resize(self, *_) -> None:
        self._build_boxes()

    def _box_rects(self) -> list[tuple[int, int, int, int]]:
        """Return (x1, y1, x2, y2) for each of the 4 boxes, centred in each quadrant."""
        sw = self.canvas.winfo_width()
        sh = self.canvas.winfo_height()
        if sw < 2 or sh < 2:
            sw, sh = self.root.winfo_width(), self.root.winfo_height()

        bw = int(sw * BOX_FRACTION)
        bh = int(sh * BOX_FRACTION)

        # Quadrant centres: each quadrant is sw/2 × sh/2
        qw, qh = sw // 2, sh // 2
        rects = []
        for col, row in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            cx = qw * col + qw // 2
            cy = qh * row + qh // 2
            rects.append((cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2))
        return rects

    def _font_size(self) -> int:
        sh = self.canvas.winfo_height() or self.root.winfo_height()
        return max(12, int(sh * 0.030))

    def _build_boxes(self) -> None:
        self.canvas.delete("all")
        self._rects = []
        self._labels = []

        font = ("Helvetica", self._font_size(), "bold")
        for i, coords in enumerate(self._box_rects()):
            x1, y1, x2, y2 = coords
            rect = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill="black", outline="#222222", width=2,
            )
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            label = self.canvas.create_text(
                cx, cy,
                text=BOXES[i]["label"],
                font=font,
                fill="#555555",
            )
            self._rects.append(rect)
            self._labels.append(label)

    # ------------------------------------------------------------------
    # Flicker loop
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now

        detected = self.state.get_detected()

        for i, box in enumerate(BOXES):
            self._phases[i] = (self._phases[i] + box["freq"] * dt) % 1.0
            on = self._phases[i] < 0.5  # square wave

            if detected is not None and abs(detected - box["freq"]) <= DETECT_TOLERANCE_HZ:
                fill = box["color"] if on else "#111111"
                text_color = "#000000" if on else box["color"]
            else:
                fill = "white" if on else "black"
                text_color = "black" if on else "#555555"

            self.canvas.itemconfig(self._rects[i], fill=fill)
            self.canvas.itemconfig(self._labels[i], fill=text_color)

        self.root.after(8, self._tick)  # ~120 Hz update loop


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSVEP 4-box flicker with BrainFlow detection.")
    parser.add_argument(
        "--board-id",
        type=int,
        default=int(BoardIds.SYNTHETIC_BOARD),
        help="BrainFlow board ID (default: synthetic board).",
    )
    parser.add_argument(
        "--no-eeg",
        action="store_true",
        help="Run visual stimulus only, no EEG acquisition.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    state = FlickerState()
    stop_event = threading.Event()

    eeg_thread = None
    if not args.no_eeg:
        eeg_thread = threading.Thread(
            target=eeg_analysis_thread,
            args=(state, args.board_id, stop_event),
            daemon=True,
        )
        eeg_thread.start()

    root = tk.Tk()
    app = SSVEPFlickerApp(root, state)  # noqa: F841

    try:
        root.mainloop()
    finally:
        stop_event.set()
        if eeg_thread is not None:
            eeg_thread.join(timeout=3.0)


if __name__ == "__main__":
    main()
