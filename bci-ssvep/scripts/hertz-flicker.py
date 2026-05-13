"""SSVEP 2-box flicker stimulus with real-time BrainFlow frequency detection.

Two boxes flicker at maximally separated SSVEP frequencies (exact 60 Hz divisors).
When the detected dominant frequency matches a box with sufficient SNR, that box
turns its highlight color.

  Box 0 (left)  :  6 Hz  -> red    (60÷10)
  Box 1 (right) : 15 Hz  -> yellow (60÷4)

Press F11 to toggle fullscreen, Escape to exit fullscreen.

Usage:
  python hertz-flicker.py --serial-port COM3 --channels 8
  python hertz-flicker.py --no-eeg   (visual only, no board needed)
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
    {"freq": 15.0, "label": "15 Hz", "color": "#FFFF44"},  # yellow (60÷4)
]

DETECT_TOLERANCE_HZ = 0.6  # peak must be within this of a target freq
SNR_THRESHOLD       = 1.5  # peak must be this many times the noise floor
WINDOW_SECONDS      = 2.0  # EEG analysis window length
ANALYSIS_INTERVAL_S = 0.5  # how often to run detection
BOX_FRACTION        = 0.38 # each box is this fraction of screen width/height


# ---------------------------------------------------------------------------
# Flicker state (shared between EEG thread and GUI thread)
# ---------------------------------------------------------------------------

class FlickerState:
    def __init__(self) -> None:
        self.detected_freq: Optional[float] = None
        self.counts: dict[float, int] = {box["freq"]: 0 for box in BOXES}
        self.total: int = 0
        self._raw_peak: Optional[float] = None
        self._raw_snr: float = 0.0
        self.lock = threading.Lock()

    def set_detected(self, freq: Optional[float], peak: float, snr: float) -> None:
        with self.lock:
            self.detected_freq = freq
            self._raw_peak = peak
            self._raw_snr = snr
            self.total += 1
            if freq is not None and freq in self.counts:
                self.counts[freq] += 1

    def get_raw(self) -> tuple[Optional[float], float]:
        with self.lock:
            return self._raw_peak, self._raw_snr

    def snapshot(self) -> tuple[Optional[float], dict[float, int], int]:
        with self.lock:
            return self.detected_freq, dict(self.counts), self.total


# ---------------------------------------------------------------------------
# EEG analysis thread
# ---------------------------------------------------------------------------

def eeg_analysis_thread(
    state: FlickerState,
    stop_event: threading.Event,
    serial_port: str,
    num_channels: int,
) -> None:
    from scipy.signal import welch as scipy_welch

    params = BrainFlowInputParams()
    params.serial_port = serial_port
    params.other_info = '{"gain": 6}'

    board_id = BoardIds.NEUROPAWN_KNIGHT_BOARD.value
    board = BoardShim(board_id, params)
    BoardShim.disable_board_logger()

    try:
        board.prepare_session()
        board.start_stream(450000)
        print("Stream started, activating channels...", flush=True)
        time.sleep(2)

        for ch in range(1, num_channels + 1):
            time.sleep(0.5)
            board.config_board(f"chon_{ch}_12")
            print(f"  chon_{ch}_12", flush=True)
            time.sleep(1)
            board.config_board(f"rldadd_{ch}")
            print(f"  rldadd_{ch}", flush=True)
            time.sleep(0.5)

        print("Board ready. Streaming EEG...", flush=True)

        fs = float(BoardShim.get_sampling_rate(board_id))
        eeg_channels = list(BoardShim.get_exg_channels(board_id))[:num_channels]
        window_samples = int(WINDOW_SECONDS * fs)

        while not stop_event.is_set():
            time.sleep(ANALYSIS_INTERVAL_S)
            if board.get_board_data_count() < window_samples:
                continue

            raw = board.get_current_board_data(window_samples)
            eeg = raw[eeg_channels, :]

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

            mask = (freqs >= 4.0) & (freqs <= 30.0)
            if not np.any(mask):
                continue

            psd_ssvep = psd_mean[mask]
            freqs_ssvep = freqs[mask]
            peak_idx = int(np.argmax(psd_ssvep))
            peak_freq = float(freqs_ssvep[peak_idx])
            peak_power = float(psd_ssvep[peak_idx])
            noise_floor = float(np.median(psd_ssvep))
            snr = peak_power / (noise_floor + 1e-12)

            matched = None
            if snr >= SNR_THRESHOLD:
                for box in BOXES:
                    if abs(peak_freq - box["freq"]) <= DETECT_TOLERANCE_HZ:
                        matched = box["freq"]
                        break

            print(f"peak={peak_freq:.2f}Hz  snr={snr:.2f}  matched={matched}", flush=True)
            state.set_detected(matched, peak_freq, snr)

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

class SSVEPFlickerApp:

    def __init__(self, root: tk.Tk, state: FlickerState) -> None:
        self.root = root
        self.state = state

        root.title("SSVEP Flicker")
        root.configure(bg="#000000")
        root.attributes("-fullscreen", True)
        root.bind("<F11>", self._toggle_fullscreen)
        root.bind("<Escape>", self._exit_fullscreen)

        self.canvas = tk.Canvas(root, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._phases = [0.0] * len(BOXES)
        self._last_time = time.perf_counter()
        self._rects: list[int] = []
        self._labels: list[int] = []
        self._status_text: Optional[int] = None
        self._fullscreen = True

        root.update_idletasks()
        self._build_layout()
        root.bind("<Configure>", lambda *_: self._build_layout())

        self._tick()

    def _toggle_fullscreen(self, *_) -> None:
        self._fullscreen = not self._fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)

    def _exit_fullscreen(self, *_) -> None:
        self._fullscreen = False
        self.root.attributes("-fullscreen", False)

    def _box_rects(self) -> list[tuple[int, int, int, int]]:
        sw = self.canvas.winfo_width() or self.root.winfo_width()
        sh = self.canvas.winfo_height() or self.root.winfo_height()
        bw = int(sw * BOX_FRACTION)
        bh = int(sh * BOX_FRACTION)
        cy = sh // 2
        rects = []
        for cx in [sw // 4, 3 * sw // 4]:
            rects.append((cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2))
        return rects

    def _build_layout(self) -> None:
        self.canvas.delete("all")
        self._rects = []
        self._labels = []
        self._status_text = None

        sw = self.canvas.winfo_width() or self.root.winfo_width()
        sh = self.canvas.winfo_height() or self.root.winfo_height()
        font = ("Helvetica", max(14, int(sh * 0.035)), "bold")

        for i, (x1, y1, x2, y2) in enumerate(self._box_rects()):
            rect = self.canvas.create_rectangle(
                x1, y1, x2, y2, fill="black", outline="#222222", width=2,
            )
            label = self.canvas.create_text(
                (x1 + x2) // 2, (y1 + y2) // 2,
                text=BOXES[i]["label"], font=font, fill="#555555",
            )
            self._rects.append(rect)
            self._labels.append(label)

        status_font = ("Courier", max(11, int(sh * 0.020)), "normal")
        self._status_text = self.canvas.create_text(
            sw // 2, sh - max(16, int(sh * 0.04)),
            text="waiting for EEG...",
            font=status_font, fill="#ffffff", anchor="center",
        )

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now

        detected, counts, total = self.state.snapshot()
        raw_peak, raw_snr = self.state.get_raw()

        for i, box in enumerate(BOXES):
            self._phases[i] = (self._phases[i] + box["freq"] * dt) % 1.0
            on = self._phases[i] < 0.5

            if detected is not None and abs(detected - box["freq"]) <= DETECT_TOLERANCE_HZ:
                fill = box["color"] if on else "#111111"
                text_color = "#000000" if on else box["color"]
            else:
                fill = "white" if on else "black"
                text_color = "black" if on else "#555555"

            self.canvas.itemconfig(self._rects[i], fill=fill)
            self.canvas.itemconfig(self._labels[i], fill=text_color)

        if self._status_text is not None:
            now_str = f"{detected:.1f} Hz" if detected is not None else "none"
            counts_str = "  |  ".join(
                f"{box['freq']:.0f}Hz: {counts.get(box['freq'], 0)}" for box in BOXES
            )
            raw_str = f"peak={raw_peak:.1f}Hz snr={raw_snr:.2f}" if raw_peak is not None else "peak=?"
            self.canvas.itemconfig(
                self._status_text,
                text=f"detected: {now_str}  ({raw_str})    [{counts_str}]    windows: {total}",
                fill="#ffffff",
            )

        self.root.after(8, self._tick)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SSVEP 2-box flicker — Neuropawn Knight.")
    parser.add_argument("--serial-port", type=str, default="COM3",
                        help="Serial port for the Neuropawn Knight (default: COM3).")
    parser.add_argument("--channels", type=int, default=8,
                        help="Number of EEG channels to activate (default: 8).")
    parser.add_argument("--no-eeg", action="store_true",
                        help="Run visual stimulus only, no board needed.")
    args = parser.parse_args()

    state = FlickerState()
    stop_event = threading.Event()

    eeg_thread = None
    if not args.no_eeg:
        eeg_thread = threading.Thread(
            target=eeg_analysis_thread,
            args=(state, stop_event, args.serial_port, args.channels),
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
            eeg_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
