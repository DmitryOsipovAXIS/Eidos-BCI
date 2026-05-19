"""SSVEP training data collection (v2) with frame-locked flicker.

Usage:
  python scripts/collect_training_data_v2.py --serial-port COM4
  python scripts/collect_training_data_v2.py --no-eeg
  python scripts/collect_training_data_v2.py --fullscreen

Notes:
  - Uses per-frame phase accumulation (not time delta) to keep flicker stable.
  - Optional single-target flicker to reduce distraction.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from utils.config import SSVEPConfig

DEFAULT_MONITOR_HZ = 60
RECORD_S = 4.0
REST_S = 2.0
CUE_S = 0.6
ITI_S = 1.0
DISCARD_S = 0.5

BG = (0, 0, 0)
WHITE = (255, 255, 255)
DIM = (20, 20, 20)
GREY = (180, 180, 180)
BLUE = (80, 120, 255)
RED = (255, 80, 80)


def _record(stream, label: int, result: dict, start_event: threading.Event) -> None:
    from acquisition.brainflow_stream import collect_labeled_window
    try:
        start_event.wait(timeout=RECORD_S + 2)
        eeg, lbl = collect_labeled_window(stream, label, RECORD_S, DISCARD_S)
        result["eeg"] = eeg
        result["label"] = lbl
    except Exception as e:
        result["error"] = str(e)


def _set_mode(size: tuple[int, int], fullscreen: bool) -> pygame.Surface:
    flags = pygame.SCALED
    if fullscreen:
        flags |= pygame.FULLSCREEN
    try:
        return pygame.display.set_mode(size, flags, vsync=1)
    except TypeError:
        return pygame.display.set_mode(size, flags)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-port", default="COM4")
    parser.add_argument("--trials", type=int, default=20, help="Total trials (left+right)")
    parser.add_argument("--no-eeg", action="store_true", help="Visual test, no board")
    parser.add_argument("--fullscreen", action="store_true", help="Use fullscreen window")
    parser.add_argument("--monitor-hz", type=float, default=DEFAULT_MONITOR_HZ,
                        help="Monitor refresh rate (default 60)")
    parser.add_argument("--left-hz", type=float, default=10.0, help="Left box frequency")
    parser.add_argument("--right-hz", type=float, default=15.0, help="Right box frequency")
    parser.add_argument("--num-channels", type=int, default=4, help="Number of EEG channels")
    parser.add_argument("--single-target-flicker", action="store_true",
                        help="Only the target flickers (default)")
    parser.add_argument("--both-flicker", action="store_true",
                        help="Both targets flicker")
    args = parser.parse_args()

    single_target = True
    if args.both_flicker:
        single_target = False

    config = SSVEPConfig(left_hz=args.left_hz, right_hz=args.right_hz)

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
        stream = BrainFlowStream(
            board_id=BoardIds.NEUROPAWN_KNIGHT_BOARD.value,
            params=p,
            num_channels=args.num_channels,
        )
        stream.prepare_session()
        stream.start_stream()
        time.sleep(2)
        for ch in range(1, 9):
            time.sleep(0.5)
            stream.board.config_board(f"chon_{ch}_12")
            time.sleep(1)
            stream.board.config_board(f"rldadd_{ch}")
            time.sleep(0.5)
        print("Board ready.", flush=True)

    # ------------------------------------------------------------------
    # pygame init
    # ------------------------------------------------------------------
    pygame.init()
    info = pygame.display.Info()
    if args.fullscreen:
        W, H = info.current_w, info.current_h
    else:
        W, H = int(info.current_w * 0.9), int(info.current_h * 0.9)
    screen = _set_mode((W, H), args.fullscreen)
    pygame.display.set_caption("SSVEP Training v2")
    clock = pygame.time.Clock()

    fbig = pygame.font.SysFont("Arial", max(36, H // 16), bold=True)
    fmed = pygame.font.SysFont("Arial", max(22, H // 28))

    bw = int(W * 0.32)
    bh = int(H * 0.42)
    cy = H // 2
    lrect = pygame.Rect(int(W * 0.08), cy - bh // 2, bw, bh)
    rrect = pygame.Rect(int(W * 0.60), cy - bh // 2, bw, bh)

    def blit_c(text: str, color: tuple, big: bool, x: int, y: int) -> None:
        s = (fbig if big else fmed).render(text, True, color)
        screen.blit(s, s.get_rect(center=(x, y)))

    def draw_fixation() -> None:
        pygame.draw.line(screen, GREY, (W // 2 - 24, H // 2), (W // 2 + 24, H // 2), 3)
        pygame.draw.line(screen, GREY, (W // 2, H // 2 - 24), (W // 2, H // 2 + 24), 3)

    def draw_arrow(cx: int, cy_: int, side: str, color: tuple) -> None:
        sz = 38
        pts = ([(cx + sz, cy_ - sz // 2), (cx - sz, cy_), (cx + sz, cy_ + sz // 2)]
               if side == "LEFT"
               else [(cx - sz, cy_ - sz // 2), (cx + sz, cy_), (cx - sz, cy_ + sz // 2)])
        pygame.draw.polygon(screen, color, pts)

    # ------------------------------------------------------------------
    # Trial sequence: interleaved LEFT / RIGHT
    # ------------------------------------------------------------------
    per_class = max(1, args.trials // 2)
    labels = [v for _ in range(per_class) for v in [0, 1]]
    if args.trials % 2 == 1:
        labels.append(0)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    running = True

    # Per-frame phase accumulator
    left_phase = 0.0
    right_phase = 0.0
    left_step = config.left_hz / args.monitor_hz
    right_step = config.right_hz / args.monitor_hz

    for idx, label in enumerate(labels):
        if not running:
            break

        side = "LEFT" if label == 0 else "RIGHT"
        col = BLUE if label == 0 else RED
        acx = lrect.centerx if label == 0 else rrect.centerx
        acy = cy - bh // 2 - 55

        # REST
        end = pygame.time.get_ticks() + int(REST_S * 1000)
        while pygame.time.get_ticks() < end:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False
            screen.fill(BG)
            blit_c(f"Trial {idx + 1} / {len(labels)}", GREY, False, W // 2, H // 8)
            draw_fixation()
            pygame.display.flip()
            clock.tick(args.monitor_hz)

        if not running:
            break

        # CUE
        end = pygame.time.get_ticks() + int(CUE_S * 1000)
        while pygame.time.get_ticks() < end:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False
            screen.fill(BG)
            blit_c(f"LOOK  {side}", col, True, W // 2, H // 8)
            draw_arrow(acx, acy, side, col)
            pygame.display.flip()
            clock.tick(args.monitor_hz)

        if not running:
            break

        # FLICKER + RECORD
        result: dict = {}
        start_event = threading.Event()
        if stream is not None:
            t = threading.Thread(
                target=_record,
                args=(stream, label, result, start_event),
                daemon=True,
            )
            t.start()
        else:
            t = None

        end = pygame.time.get_ticks() + int(RECORD_S * 1000)
        frames = 0
        started = False
        while pygame.time.get_ticks() < end:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False

            left_phase = (left_phase + left_step) % 1.0
            right_phase = (right_phase + right_step) % 1.0
            left_on = left_phase < 0.5
            right_on = right_phase < 0.5

            screen.fill(BG)
            blit_c(f"FOCUS  {side}", col, True, W // 2, H // 8)

            if single_target:
                left_color = WHITE if (label == 0 and left_on) else DIM
                right_color = WHITE if (label == 1 and right_on) else DIM
            else:
                left_color = WHITE if left_on else DIM
                right_color = WHITE if right_on else DIM

            pygame.draw.rect(screen, left_color, lrect)
            pygame.draw.rect(screen, right_color, rrect)
            pygame.draw.rect(screen, BLUE if label == 0 else (40, 40, 40), lrect, 4)
            pygame.draw.rect(screen, RED if label == 1 else (40, 40, 40), rrect, 4)

            blit_c(f"{config.left_hz:.0f} Hz", BLUE, False, lrect.centerx, lrect.bottom + 20)
            blit_c(f"{config.right_hz:.0f} Hz", RED, False, rrect.centerx, rrect.bottom + 20)

            fps = clock.get_fps()
            blit_c(f"FPS {fps:0.1f}", GREY, False, W - 90, 26)

            pygame.display.flip()
            if stream is not None and not started:
                start_event.set()
                started = True
            clock.tick(args.monitor_hz)
            frames += 1

        if t is not None:
            t.join(timeout=RECORD_S + 2)
            if "error" in result:
                print(f"  Trial {idx + 1} error: {result['error']}", flush=True)
            elif "eeg" in result:
                X_list.append(result["eeg"])
                y_list.append(result["label"])
                print(f"  Trial {idx + 1}  {side}  {result['eeg'].shape}", flush=True)

        # ITI
        end = pygame.time.get_ticks() + int(ITI_S * 1000)
        while pygame.time.get_ticks() < end:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False
            screen.fill(BG)
            pygame.display.flip()
            clock.tick(args.monitor_hz)

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
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ml = min(x.shape[1] for x in X_list)
        X = np.stack([x[:, :ml] for x in X_list])
        y = np.array(y_list, dtype=np.int64)
        np.save(config.data_raw_dir / f"X_{ts}.npy", X)
        np.save(config.data_raw_dir / f"y_{ts}.npy", y)
        print(f"\nSaved {X.shape[0]} trials  X={X.shape}  "
              f"LEFT={(y == 0).sum()}  RIGHT={(y == 1).sum()}")
    elif args.no_eeg:
        print("\n--no-eeg: visual test done, no data saved.")
    else:
        print("\nNo trials completed -- nothing saved.")


if __name__ == "__main__":
    main()
