"""Real-time SSVEP topographic map for the Neuropawn Knight.

Shows a bird's-eye head diagram with the 8 electrodes (PO3 POz PO4 PO7 O7 Oz O2 PO8)
coloured by their individual SSVEP band power at the currently detected frequency.
Brighter/warmer = stronger signal at that electrode.

Run alongside hertz-flicker.py, or standalone to check signal quality.

Usage:
  python scripts/topomap.py --serial-port COM4 --channels 8
  python scripts/topomap.py --no-eeg   (random demo data)
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.signal import butter, filtfilt, welch as scipy_welch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

# ---------------------------------------------------------------------------
# Electrode layout — your actual Knight channel positions
# Coordinates in normalised head space: (0,0) = centre, 1.0 = head edge
# ---------------------------------------------------------------------------

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

TARGET_FREQS = [12.0, 15.0]   # must match hertz-flicker.py
BAND_HW      = 0.8            # ±Hz around each target
HIGHPASS_HZ  = 11.0
WINDOW_S     = 4.0
UPDATE_S     = 0.5


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class TopoState:
    def __init__(self, n_ch: int) -> None:
        self.powers   = np.zeros((len(TARGET_FREQS), n_ch))  # (n_freqs, n_ch)
        self.detected: Optional[float] = None
        self.lock = threading.Lock()

    def update(self, powers: np.ndarray, detected: Optional[float]) -> None:
        with self.lock:
            self.powers[:] = powers
            self.detected = detected

    def snapshot(self) -> tuple[np.ndarray, Optional[float]]:
        with self.lock:
            return self.powers.copy(), self.detected


# ---------------------------------------------------------------------------
# EEG thread
# ---------------------------------------------------------------------------

def eeg_thread_fn(state: TopoState, serial_port: str, num_channels: int,
                  stop: threading.Event) -> None:
    params = BrainFlowInputParams()
    params.serial_port = serial_port
    board_id = BoardIds.NEUROPAWN_KNIGHT_BOARD.value
    board = BoardShim(board_id, params)
    BoardShim.disable_board_logger()

    try:
        board.prepare_session()
        board.start_stream(450000)
        time.sleep(2)
        for ch in range(1, num_channels + 1):
            time.sleep(0.5)
            board.config_board(f"chon_{ch}_12")
            time.sleep(1)
            board.config_board(f"rldadd_{ch}")
            time.sleep(0.5)

        fs = float(BoardShim.get_sampling_rate(board_id))
        eeg_idx = list(BoardShim.get_exg_channels(board_id))[:num_channels]
        win = int(WINDOW_S * fs)
        nperseg = min(int(fs * 2), win)
        b_hp, a_hp = butter(4, HIGHPASS_HZ / (fs / 2), btype='high')

        while not stop.is_set():
            time.sleep(UPDATE_S)
            if board.get_board_data_count() < win:
                continue

            raw = board.get_current_board_data(win)
            eeg = raw[eeg_idx, :]  # (n_ch, samples)
            eeg = filtfilt(b_hp, a_hp, eeg, axis=1)

            # Per-channel PSD
            ch_powers = np.zeros((len(TARGET_FREQS), len(eeg_idx)))
            baseline_per_ch = np.zeros(len(eeg_idx))

            for ci in range(len(eeg_idx)):
                f, pxx = scipy_welch(eeg[ci], fs=fs, nperseg=nperseg,
                                     noverlap=nperseg // 2)
                base_mask = (f >= 8.0) & (f <= 30.0)
                baseline_per_ch[ci] = float(np.median(pxx[base_mask])) + 1e-12
                for fi, freq in enumerate(TARGET_FREQS):
                    band = (f >= freq - BAND_HW) & (f <= freq + BAND_HW)
                    ch_powers[fi, ci] = float(np.max(pxx[band])) if np.any(band) else 0.0

            # SNR per channel per freq
            snr = ch_powers / baseline_per_ch[np.newaxis, :]

            # Detected freq = whichever target has highest mean SNR across channels
            mean_snr = snr.mean(axis=1)
            best = int(np.argmax(mean_snr))
            detected = TARGET_FREQS[best] if mean_snr[best] >= 1.5 else None

            state.update(snr, detected)

    except Exception as e:
        print(f"EEG error: {e}", flush=True)
        import traceback; traceback.print_exc()
    finally:
        try: board.stop_stream()
        except Exception: pass
        try: board.release_session()
        except Exception: pass


def demo_thread_fn(state: TopoState, stop: threading.Event) -> None:
    """Generate fake data for --no-eeg mode."""
    rng = np.random.default_rng(0)
    while not stop.is_set():
        time.sleep(UPDATE_S)
        snr = rng.uniform(0.8, 2.5, (len(TARGET_FREQS), len(CHANNELS)))
        snr[0, 5:] *= 3   # make PO3/POz/PO4 brighter for freq 0
        detected = TARGET_FREQS[0]
        state.update(snr, detected)


# ---------------------------------------------------------------------------
# Matplotlib topographic plot
# ---------------------------------------------------------------------------

def build_plot(state: TopoState) -> None:
    n_freqs = len(TARGET_FREQS)
    fig, axes = plt.subplots(1, n_freqs, figsize=(4 * n_freqs, 5))
    fig.patch.set_facecolor("#111111")
    if n_freqs == 1:
        axes = [axes]

    FREQ_COLORS = ["#FF6666", "#FFFF66"]
    EL_RADIUS = 0.09
    HEAD_RADIUS = 0.72

    # Pre-build artists so we can update without redrawing everything
    el_circles: list[list[plt.Circle]] = []
    title_texts: list[plt.Text] = []

    for fi, (ax, freq) in enumerate(zip(axes, TARGET_FREQS)):
        ax.set_facecolor("#111111")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.axis("off")

        # Head outline
        head = plt.Circle((0, 0), HEAD_RADIUS, color="#333333", fill=False, lw=2)
        ax.add_patch(head)

        # Nose
        ax.annotate("", xy=(0, HEAD_RADIUS + 0.08), xytext=(0, HEAD_RADIUS),
                    arrowprops=dict(arrowstyle="-", color="#444444", lw=2))

        # Ears
        for ex in [-HEAD_RADIUS, HEAD_RADIUS]:
            ear = mpatches.FancyBboxPatch(
                (ex - 0.04, -0.12), 0.08, 0.24,
                boxstyle="round,pad=0.02", color="#333333", zorder=0)
            ax.add_patch(ear)

        # Electrode circles (start dark, updated each frame)
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

        t = ax.set_title(f"{freq:.0f} Hz", color=FREQ_COLORS[fi],
                         fontsize=14, fontweight="bold", pad=8)
        title_texts.append(t)

    # Colorbar
    cmap = plt.cm.hot
    norm = Normalize(vmin=1.0, vmax=6.0)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("SNR", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    status_ax = fig.add_axes([0.0, 0.0, 1.0, 0.04])
    status_ax.set_facecolor("#111111")
    status_ax.axis("off")
    status_txt = status_ax.text(0.5, 0.5, "waiting...",
                                ha="center", va="center",
                                color="white", fontsize=10,
                                transform=status_ax.transAxes)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.canvas.manager.set_window_title("SSVEP Topography")

    cmap_fn = plt.cm.hot

    def animate(_frame=None):
        powers, detected = state.snapshot()
        for fi in range(n_freqs):
            for ci, circle in enumerate(el_circles[fi]):
                snr_val = float(powers[fi, ci])
                rgba = cmap_fn(norm(np.clip(snr_val, 1.0, 6.0)))
                circle.set_facecolor(rgba)

            is_detected = detected is not None and abs(detected - TARGET_FREQS[fi]) < 0.5
            title_texts[fi].set_color("#ffffff" if is_detected else FREQ_COLORS[fi])

        det_str = f"detected: {detected:.0f} Hz" if detected else "detected: none"
        status_txt.set_text(det_str)
        fig.canvas.draw_idle()

    # Use matplotlib's timer for live updates
    timer = fig.canvas.new_timer(interval=int(UPDATE_S * 1000))
    timer.add_callback(animate)
    timer.start()

    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time SSVEP topographic map.")
    parser.add_argument("--serial-port", type=str, default="COM4")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--no-eeg", action="store_true",
                        help="Demo mode with fake data.")
    args = parser.parse_args()

    state = TopoState(args.channels)
    stop = threading.Event()

    if args.no_eeg:
        t = threading.Thread(target=demo_thread_fn, args=(state, stop), daemon=True)
    else:
        t = threading.Thread(target=eeg_thread_fn,
                             args=(state, args.serial_port, args.channels, stop),
                             daemon=True)
    t.start()

    try:
        build_plot(state)
    finally:
        stop.set()
        t.join(timeout=3.0)


if __name__ == "__main__":
    main()
