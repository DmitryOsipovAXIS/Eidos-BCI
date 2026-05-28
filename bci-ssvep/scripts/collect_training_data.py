"""SSVEP training data collection — pygame stimulus + BrainFlow recording.

Usage:
  python scripts/collect_training_data.py --serial-port COM4
  python scripts/collect_training_data.py --no-eeg   # visual test only
  python scripts/collect_training_data.py --fullscreen  !!!!

"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from utils.config import SSVEPConfig

MONITOR_HZ = 60
BLOCK_S    = 30.0
TOTAL_S    = 5 * 60.0
DISCARD_S  = 0.5

BG    = (0,   0,   0)
WHITE = (255, 255, 255)
DIM   = (20,  20,  20)
GREY  = (180, 180, 180)
BLUE  = (80,  120, 255)
RED   = (255, 80,  80)


def _record(stream, label: int, result: dict) -> None:
    from acquisition.brainflow_stream import collect_labeled_window
    try:
        eeg, lbl = collect_labeled_window(stream, label, BLOCK_S, DISCARD_S)
        result["eeg"]   = eeg
        result["label"] = lbl
    except Exception as e:
        result["error"] = str(e)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-port", default="COM4")
    parser.add_argument("--no-eeg", action="store_true", help="Visual test, no board")
    parser.add_argument("--fullscreen", action="store_true", help="Use fullscreen stimulus window")
    parser.add_argument("--left-hz",      type=float, default=7.5, help="Left box frequency (default 7.5)")
    parser.add_argument("--right-hz",     type=float, default=12.0, help="Right box frequency (default 12.0)")
    parser.add_argument("--num-channels", type=int,   default=8,    help="Number of EEG channels (default 8)")
    args = parser.parse_args()

    config = SSVEPConfig(left_hz=args.left_hz, right_hz=args.right_hz)

    def fmt_hz(value: float) -> str:
        return f"{value:0.3f} Hz"

    def print_recorded_psd(side: str, eeg: np.ndarray, fs: float) -> None:
        n_samples = eeg.shape[1]
        nperseg = min(1024, n_samples)
        psds = []
        freqs = None
        for ch in range(eeg.shape[0]):
            freqs, pxx = welch(eeg[ch], fs=fs, nperseg=nperseg)
            psds.append(pxx)
        mean_psd = np.mean(np.stack(psds, axis=0), axis=0)

        target_bins = []
        for target in (config.left_hz, config.right_hz):
            idx = int(np.argmin(np.abs(freqs - target)))
            target_bins.append((target, float(freqs[idx]), float(mean_psd[idx])))

        mask = (freqs >= max(1.0, min(config.left_hz, config.right_hz) - 2.0)) & \
               (freqs <= max(config.left_hz, config.right_hz) + 2.0)
        peak_freq = float(freqs[mask][np.argmax(mean_psd[mask])]) if np.any(mask) else float(freqs[np.argmax(mean_psd)])
        peak_power = float(np.max(mean_psd[mask])) if np.any(mask) else float(np.max(mean_psd))

        print(f"{side}: recorded EEG shape={eeg.shape}", flush=True)
        print(f"{side}: peak frequency in target band={peak_freq:0.3f} Hz (power={peak_power:0.3g})", flush=True)
        for target, bin_freq, bin_power in target_bins:
            print(f"{side}: power near {target:0.3f} Hz -> bin {bin_freq:0.3f} Hz, power={bin_power:0.3g}", flush=True)

    print("Configured stimulus labels:", flush=True)
    print("LEFT:  label=0", flush=True)
    print("RIGHT: label=1", flush=True)
    print("Console will show PSD summaries from the recorded EEG that gets saved to disk.", flush=True)
    print(f"block duration={BLOCK_S:0.1f}s, total flicker time={TOTAL_S/60:0.1f} min, discard start={DISCARD_S:0.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Board init
    # ------------------------------------------------------------------
    stream = None
    if not args.no_eeg:
        from acquisition.brainflow_stream import BrainFlowStream
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        BoardShim.disable_board_logger()
        p = BrainFlowInputParams()
        p.serial_port = args.serial_port
        stream = BrainFlowStream(board_id=BoardIds.NEUROPAWN_KNIGHT_BOARD.value, params=p,
                                 num_channels=args.num_channels)
        stream.prepare_session()
        stream.start_stream()
        actual_fs = stream.sampling_rate()
        if abs(actual_fs - 125.0) > 1e-6:
            stream.stop_stream()
            stream.release_session()
            raise RuntimeError(f"Board sampling rate is {actual_fs:0.3f} Hz, expected 125.000 Hz")
        time.sleep(2)
        for ch in range(1, 9):
            time.sleep(0.5); stream.board.config_board(f"chon_{ch}_12")
            time.sleep(1);   stream.board.config_board(f"rldadd_{ch}")
            time.sleep(0.5)
        print(f"Board ready at {actual_fs:0.3f} Hz.", flush=True)

    # ------------------------------------------------------------------
    # pygame init
    # ------------------------------------------------------------------
    pygame.init()
    info = pygame.display.Info()
    if args.fullscreen:
        W, H = info.current_w, info.current_h
        screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
    else:
        W, H = int(info.current_w * 0.9), int(info.current_h * 0.9)
        screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    pygame.display.set_caption("SSVEP Training")
    clock = pygame.time.Clock()

    fbig = pygame.font.SysFont("Arial", max(36, H // 16), bold=True)
    fmed = pygame.font.SysFont("Arial", max(22, H // 28))

    bw = int(W * 0.30)
    bh = int(H * 0.40)
    cy = H // 2
    lrect = pygame.Rect(int(W * 0.10), cy - bh // 2, bw, bh)
    rrect = pygame.Rect(int(W * 0.60), cy - bh // 2, bw, bh)

    def blit_c(text: str, color: tuple, big: bool, x: int, y: int) -> None:
        s = (fbig if big else fmed).render(text, True, color)
        screen.blit(s, s.get_rect(center=(x, y)))

    def draw_arrow(cx: int, cy_: int, side: str, color: tuple) -> None:
        sz = 38
        pts = ([(cx+sz, cy_-sz//2), (cx-sz, cy_), (cx+sz, cy_+sz//2)] if side == "LEFT"
               else [(cx-sz, cy_-sz//2), (cx+sz, cy_), (cx-sz, cy_+sz//2)])
        pygame.draw.polygon(screen, color, pts)

    # ------------------------------------------------------------------
    # Trial sequence: interleaved LEFT / RIGHT for 5 minutes total
    # ------------------------------------------------------------------
    labels = [0 if i % 2 == 0 else 1 for i in range(int(TOTAL_S / BLOCK_S))]
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    running = True

    print(f"Planned schedule: {len(labels)} blocks x {BLOCK_S:0.1f}s = {TOTAL_S/60:0.1f} min", flush=True)
    print("Alternation: LEFT, RIGHT, LEFT, RIGHT, ...", flush=True)

    for idx, label in enumerate(labels):
        if not running:
            break

        side = "LEFT" if label == 0 else "RIGHT"
        col  = BLUE   if label == 0 else RED
        acx  = lrect.centerx if label == 0 else rrect.centerx
        acy  = cy - bh // 2 - 55

        print(f"\nTRIAL {idx + 1:02d}/{len(labels):02d}", flush=True)
        print(f"{side}: label={label}", flush=True)

        # FLICKER + RECORD
        result: dict = {}
        if stream is not None:
            t = threading.Thread(target=_record, args=(stream, label, result), daemon=True)
            t.start()
        else:
            t = None

        left_phase  = 0.0
        right_phase = 0.0
        last_t = time.perf_counter()
        block_start = time.perf_counter()
        block_end = block_start + BLOCK_S
        while time.perf_counter() < block_end:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False

            now = time.perf_counter()
            dt  = now - last_t
            last_t = now
            left_phase  = (left_phase  + config.left_hz  * dt) % 1.0
            right_phase = (right_phase + config.right_hz * dt) % 1.0

            left_on  = left_phase  < 0.5
            right_on = right_phase < 0.5

            screen.fill(BG)
            blit_c(f"FOCUS  {side}", col, True, W // 2, H // 8)

            pygame.draw.rect(screen, WHITE if left_on  else DIM, lrect)
            pygame.draw.rect(screen, WHITE if right_on else DIM, rrect)
            pygame.draw.rect(screen, BLUE if label == 0 else (40, 40, 40), lrect,  4)
            pygame.draw.rect(screen, RED  if label == 1 else (40, 40, 40), rrect, 4)

            blit_c(f"{config.left_hz:.1f} Hz",  BLUE, False, lrect.centerx, lrect.bottom + 20)
            blit_c(f"{config.right_hz:.1f} Hz", RED,  False, rrect.centerx, rrect.bottom + 20)
            blit_c(f"Block {idx + 1}/{len(labels)}  |  {max(0.0, block_end - now):0.1f}s", GREY, False, W // 2, H // 8)

            pygame.display.flip()
            clock.tick(MONITOR_HZ)

        if t is not None:
            t.join(timeout=BLOCK_S + 2)
            if "error" in result:
                print(f"  Trial {idx+1} error: {result['error']}", flush=True)
            elif "eeg" in result:
                X_list.append(result["eeg"])
                y_list.append(result["label"])
                print(f"  Trial {idx+1}  {side}  {result['eeg'].shape}", flush=True)
                print_recorded_psd(side, result["eeg"], stream.sampling_rate())
        elif args.no_eeg:
            print(f"{side}: no EEG recorded in --no-eeg mode", flush=True)

    pygame.quit()

    # ------------------------------------------------------------------
    # Teardown board
    # ------------------------------------------------------------------
    if stream is not None:
        try:
            stream.stop_stream()
        except Exception:
            pass
        stream.release_session()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if X_list:
        config.data_raw_dir.mkdir(parents=True, exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        ml  = min(x.shape[1] for x in X_list)
        X   = np.stack([x[:, :ml] for x in X_list])
        y   = np.array(y_list, dtype=np.int64)
        print("\nSaving raw data:", flush=True)
        print(f"  X shape: {X.shape}", flush=True)
        print(f"  y shape: {y.shape}", flush=True)
        print(f"  LEFT label=0 target={config.left_hz:0.3f} Hz", flush=True)
        print(f"  RIGHT label=1 target={config.right_hz:0.3f} Hz", flush=True)
        print(f"  output files: X_{ts}.npy / y_{ts}.npy", flush=True)
        np.save(config.data_raw_dir / f"X_{ts}.npy", X)
        np.save(config.data_raw_dir / f"y_{ts}.npy", y)
        print(f"\nSaved {X.shape[0]} trials  X={X.shape}  "
              f"LEFT={(y==0).sum()}  RIGHT={(y==1).sum()}")
    elif args.no_eeg:
        print("\n--no-eeg: visual test done, no data saved.")
    else:
        print("\nNo trials completed — nothing saved.")


if __name__ == "__main__":
    main()
