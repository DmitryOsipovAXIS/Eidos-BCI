"""SSVEP flicker stimulus + real-time topographic map — Neuropawn Knight.

Two windows open simultaneously sharing one board connection:
  1. Fullscreen flicker: two boxes at 12 Hz (red) and 15 Hz (yellow)
  2. Topomap: bird's-eye head showing per-channel SNR at each target frequency

Electrodes: PO3 POz PO4 PO7 O7 Oz O2 PO8

Usage:
  python hertz-flicker.py --serial-port COM4 --channels 8
  python hertz-flicker.py --no-eeg   (visual + demo topomap, no board needed)
  Press F11 to toggle fullscreen, Escape to exit fullscreen.
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
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.signal import butter, filtfilt, welch as scipy_welch

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOXES = [
    {"freq": 12.0, "label": "12 Hz", "color": "#FF4444"},  # red    (60÷5)
    {"freq": 15.0, "label": "15 Hz", "color": "#FFFF44"},  # yellow (60÷4)
]

CHANNELS = [
    {"name": "PO3", "pos": (-0.28, -0.18)},
    {"name": "POz", "pos": ( 0.00, -0.22)},
    {"name": "PO4", "pos": ( 0.28, -0.18)},
    {"name": "PO7", "pos": (-0.52, -0.22)},
    {"name": "O7",  "pos": (-0.42, -0.48)},
    {"name": "Oz",  "pos": ( 0.00, -0.55)},
    {"name": "O2",  "pos": ( 0.28, -0.48)},
    {"name": "PO8", "pos": ( 0.52, -0.22)},
]

DETECT_TOLERANCE_HZ = 0.8
SNR_THRESHOLD       = 1.5
WINDOW_SECONDS      = 4.0
ANALYSIS_INTERVAL_S = 0.5
HIGHPASS_HZ         = 11.0
BOX_FRACTION        = 0.38


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class SharedState:
    def __init__(self) -> None:
        self.detected_freq: Optional[float] = None
        self.counts: dict[float, int] = {box["freq"]: 0 for box in BOXES}
        self.total: int = 0
        self._raw_peak: Optional[float] = None
        self._raw_snr: float = 0.0
        # per-channel SNR: shape (n_freqs, n_channels)
        self.ch_snr: np.ndarray = np.zeros((len(BOXES), len(CHANNELS)))
        self.lock = threading.Lock()

    def update(self, detected: Optional[float], peak: float, snr: float,
               ch_snr: np.ndarray) -> None:
        with self.lock:
            self.detected_freq = detected
            self._raw_peak = peak
            self._raw_snr = snr
            self.ch_snr = ch_snr.copy()
            self.total += 1
            if detected is not None and detected in self.counts:
                self.counts[detected] += 1

    def snapshot(self) -> tuple[Optional[float], dict[float, int], int, Optional[float], float]:
        with self.lock:
            return (self.detected_freq, dict(self.counts), self.total,
                    self._raw_peak, self._raw_snr)

    def ch_snr_snapshot(self) -> tuple[np.ndarray, Optional[float]]:
        with self.lock:
            return self.ch_snr.copy(), self.detected_freq


# ---------------------------------------------------------------------------
# EEG thread
# ---------------------------------------------------------------------------

def eeg_thread_fn(state: SharedState, stop: threading.Event,
                  serial_port: str, num_channels: int) -> None:
    params = BrainFlowInputParams()
    params.serial_port = serial_port
    board_id = BoardIds.NEUROPAWN_KNIGHT_BOARD.value
    board = BoardShim(board_id, params)
    BoardShim.disable_board_logger()

    try:
        print(f"Connecting to Neuropawn Knight on {serial_port}...", flush=True)
        board.prepare_session()
        print("Session prepared. Starting stream...", flush=True)
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
        eeg_idx = list(BoardShim.get_exg_channels(board_id))[:num_channels]
        win = int(WINDOW_SECONDS * fs)
        nperseg = min(int(fs * 2), win)
        b_hp, a_hp = butter(4, HIGHPASS_HZ / (fs / 2), btype='high')

        while not stop.is_set():
            time.sleep(ANALYSIS_INTERVAL_S)
            if board.get_board_data_count() < win:
                continue

            raw = board.get_current_board_data(win)
            eeg = raw[eeg_idx, :]
            eeg = filtfilt(b_hp, a_hp, eeg, axis=1)

            # Per-channel PSD for topomap
            ch_snr = np.zeros((len(BOXES), len(eeg_idx)))
            psds, freqs = [], None
            for ci in range(len(eeg_idx)):
                f, pxx = scipy_welch(eeg[ci], fs=fs, nperseg=nperseg,
                                     noverlap=nperseg // 2)
                psds.append(pxx)
                freqs = f
                base = float(np.median(pxx[(f >= 8.0) & (f <= 30.0)])) + 1e-12
                for fi, box in enumerate(BOXES):
                    band = (f >= box["freq"] - DETECT_TOLERANCE_HZ) & \
                           (f <= box["freq"] + DETECT_TOLERANCE_HZ)
                    ch_snr[fi, ci] = float(np.max(pxx[band])) / base if np.any(band) else 0.0

            if freqs is None:
                continue

            # Mean PSD for detection
            psd_mean = np.mean(np.stack(psds, axis=0), axis=0)
            baseline = float(np.median(psd_mean[(freqs >= 8.0) & (freqs <= 30.0)])) + 1e-12

            box_snrs = []
            for box in BOXES:
                band = (freqs >= box["freq"] - DETECT_TOLERANCE_HZ) & \
                       (freqs <= box["freq"] + DETECT_TOLERANCE_HZ)
                power = float(np.max(psd_mean[band])) if np.any(band) else 0.0
                box_snrs.append(power / baseline)

            best_idx = int(np.argmax(box_snrs))
            best_snr = box_snrs[best_idx]
            best_freq = BOXES[best_idx]["freq"]
            second_snr = box_snrs[1 - best_idx]
            matched = best_freq if (best_snr >= SNR_THRESHOLD and
                                    best_snr > second_snr * 1.2) else None

            snr_str = "  ".join(f"{BOXES[i]['freq']:.0f}Hz={box_snrs[i]:.2f}"
                                for i in range(len(BOXES)))
            print(f"SNRs: {snr_str}  -> matched={matched}", flush=True)
            state.update(matched, best_freq, best_snr, ch_snr)

    except Exception as exc:
        print(f"EEG thread error: {exc}", flush=True)
        import traceback; traceback.print_exc()
    finally:
        try: board.stop_stream()
        except Exception: pass
        try: board.release_session()
        except Exception: pass


def demo_eeg_thread_fn(state: SharedState, stop: threading.Event) -> None:
    rng = np.random.default_rng(0)
    while not stop.is_set():
        time.sleep(ANALYSIS_INTERVAL_S)
        ch_snr = rng.uniform(0.8, 2.0, (len(BOXES), len(CHANNELS)))
        ch_snr[0, 4:] *= 4
        state.update(BOXES[0]["freq"], BOXES[0]["freq"], 3.0, ch_snr)


# ---------------------------------------------------------------------------
# Flicker window (tkinter)
# ---------------------------------------------------------------------------

class FlickerWindow:

    def __init__(self, root: tk.Tk, state: SharedState) -> None:
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
        bw, bh = int(sw * BOX_FRACTION), int(sh * BOX_FRACTION)
        cy = sh // 2
        return [(cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2)
                for cx in [sw // 4, 3 * sw // 4]]

    def _build_layout(self) -> None:
        self.canvas.delete("all")
        self._rects, self._labels, self._status_text = [], [], None
        sw = self.canvas.winfo_width() or self.root.winfo_width()
        sh = self.canvas.winfo_height() or self.root.winfo_height()
        font = ("Helvetica", max(14, int(sh * 0.035)), "bold")
        for i, (x1, y1, x2, y2) in enumerate(self._box_rects()):
            self._rects.append(self.canvas.create_rectangle(
                x1, y1, x2, y2, fill="black", outline="#222222", width=2))
            self._labels.append(self.canvas.create_text(
                (x1 + x2) // 2, (y1 + y2) // 2,
                text=BOXES[i]["label"], font=font, fill="#555555"))
        status_font = ("Courier", max(11, int(sh * 0.020)), "normal")
        self._status_text = self.canvas.create_text(
            sw // 2, sh - max(16, int(sh * 0.04)),
            text="waiting for EEG...", font=status_font,
            fill="#ffffff", anchor="center")

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now

        detected, counts, total, raw_peak, raw_snr = self.state.snapshot()

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
                f"{box['freq']:.0f}Hz: {counts.get(box['freq'], 0)}" for box in BOXES)
            raw_str = f"peak={raw_peak:.1f}Hz snr={raw_snr:.2f}" if raw_peak else "peak=?"
            self.canvas.itemconfig(
                self._status_text,
                text=f"detected: {now_str}  ({raw_str})    [{counts_str}]    windows: {total}",
                fill="#ffffff")

        self.root.after(8, self._tick)


# ---------------------------------------------------------------------------
# Topomap window (matplotlib)
# ---------------------------------------------------------------------------

def build_topomap(state: SharedState) -> None:
    n_freqs = len(BOXES)
    FREQ_COLORS = ["#FF6666", "#FFFF66"]
    EL_RADIUS = 0.09
    HEAD_RADIUS = 0.72
    cmap = plt.cm.hot
    norm = Normalize(vmin=1.0, vmax=6.0)

    fig, axes = plt.subplots(1, n_freqs, figsize=(4 * n_freqs, 5))
    fig.patch.set_facecolor("#111111")

    el_circles: list[list[plt.Circle]] = []
    title_texts: list[plt.Text] = []

    for fi, ax in enumerate(axes):
        ax.set_facecolor("#111111")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_aspect("equal"); ax.axis("off")

        ax.add_patch(plt.Circle((0, 0), HEAD_RADIUS,
                                color="#333333", fill=False, lw=2))
        ax.annotate("", xy=(0, HEAD_RADIUS + 0.08), xytext=(0, HEAD_RADIUS),
                    arrowprops=dict(arrowstyle="-", color="#444444", lw=2))
        for ex in [-HEAD_RADIUS, HEAD_RADIUS]:
            ax.add_patch(mpatches.FancyBboxPatch(
                (ex - 0.04, -0.12), 0.08, 0.24,
                boxstyle="round,pad=0.02", color="#333333", zorder=0))

        circles = []
        for ch in CHANNELS:
            c = plt.Circle(ch["pos"], EL_RADIUS, color="#222222",
                           zorder=3, ec="#555555", lw=1.5)
            ax.add_patch(c)
            ax.text(ch["pos"][0], ch["pos"][1] - EL_RADIUS - 0.06,
                    ch["name"], ha="center", va="top",
                    fontsize=7, color="#888888")
            circles.append(c)
        el_circles.append(circles)
        title_texts.append(ax.set_title(
            f"{BOXES[fi]['freq']:.0f} Hz", color=FREQ_COLORS[fi],
            fontsize=14, fontweight="bold", pad=8))

    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("SNR", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    status_ax = fig.add_axes([0.0, 0.0, 1.0, 0.04])
    status_ax.set_facecolor("#111111"); status_ax.axis("off")
    status_txt = status_ax.text(0.5, 0.5, "waiting...",
                                ha="center", va="center", color="white",
                                fontsize=10, transform=status_ax.transAxes)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.canvas.manager.set_window_title("SSVEP Topography")

    def animate(_=None):
        ch_snr, detected = state.ch_snr_snapshot()
        for fi in range(n_freqs):
            for ci, circle in enumerate(el_circles[fi]):
                val = float(ch_snr[fi, ci]) if ci < ch_snr.shape[1] else 1.0
                circle.set_facecolor(cmap(norm(np.clip(val, 1.0, 6.0))))
            is_det = detected is not None and abs(detected - BOXES[fi]["freq"]) < 0.5
            title_texts[fi].set_color("#ffffff" if is_det else FREQ_COLORS[fi])
        det_str = f"detected: {detected:.0f} Hz" if detected else "detected: none"
        status_txt.set_text(det_str)
        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=int(ANALYSIS_INTERVAL_S * 1000))
    timer.add_callback(animate)
    timer.start()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SSVEP flicker + topomap — Neuropawn Knight.")
    parser.add_argument("--serial-port", type=str, default="COM4")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--no-eeg", action="store_true",
                        help="Run with demo data, no board needed.")
    args = parser.parse_args()

    state = SharedState()
    stop = threading.Event()

    # EEG thread
    if args.no_eeg:
        t = threading.Thread(target=demo_eeg_thread_fn, args=(state, stop), daemon=True)
    else:
        t = threading.Thread(target=eeg_thread_fn,
                             args=(state, stop, args.serial_port, args.channels),
                             daemon=True)
    t.start()

    # Topomap in its own thread (matplotlib blocks)
    topo_thread = threading.Thread(target=build_topomap, args=(state,), daemon=True)
    topo_thread.start()

    # Flicker window on main thread
    root = tk.Tk()
    FlickerWindow(root, state)

    try:
        root.mainloop()
    finally:
        stop.set()
        t.join(timeout=5.0)


if __name__ == "__main__":
    main()
