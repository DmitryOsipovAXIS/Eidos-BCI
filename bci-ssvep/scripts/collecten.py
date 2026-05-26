"""Step 1: minimal starter for SSVEP data collection.

This file is intentionally small for a from-scratch rebuild.
We will add one piece at a time and explain each part.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time
from typing import Optional
import threading

import numpy as np
import pygame
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
MONITOR_HZ = 60
TARGET_HZ = 7.5
TEST_DURATION_S = 60.0
DISCARD_S = 0.5

import sys
if str(ROOT / "src") not in sys.path:
	sys.path.insert(0, str(ROOT / "src"))


def record_and_compute_psd(stream, duration_s: float, discard_s: float, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	from acquisition.brainflow_stream import collect_labeled_window

	eeg, _ = collect_labeled_window(stream, 0, duration_s, discard_s)
	nperseg = min(2048, eeg.shape[1])
	psds = []
	freqs = None
	for ch in range(eeg.shape[0]):
		freqs, pxx = welch(eeg[ch], fs=fs, nperseg=nperseg)
		psds.append(pxx)
	mean_psd = np.mean(np.stack(psds, axis=0), axis=0)
	return freqs, mean_psd, eeg


def visual_flicker(duration_s: float, hz: float, fullscreen: bool) -> None:
	pygame.init()
	info = pygame.display.Info()
	if fullscreen:
		W, H = info.current_w, info.current_h
		screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
	else:
		W, H = int(info.current_w * 0.9), int(info.current_h * 0.9)
		screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
	clock = pygame.time.Clock()
	fbig = pygame.font.SysFont("Arial", max(36, H // 16), bold=True)
	lrect = pygame.Rect(int(W * 0.35), H // 4, int(W * 0.3), int(H * 0.5))

	phase = 0.0
	last_t = time.perf_counter()
	end_t = last_t + duration_s
	while time.perf_counter() < end_t:
		for e in pygame.event.get():
			if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
				pygame.quit()
				return
		now = time.perf_counter()
		dt = now - last_t
		last_t = now
		phase = (phase + hz * dt) % 1.0
		on = phase < 0.5
		screen.fill((0, 0, 0))
		pygame.draw.rect(screen, (255, 255, 255) if on else (20, 20, 20), lrect)
		s = fbig.render(f"{hz:0.1f} Hz", True, (180, 180, 180))
		screen.blit(s, s.get_rect(center=(W // 2, H // 8)))
		pygame.display.flip()
		clock.tick(MONITOR_HZ)
	pygame.quit()


def main() -> None:
	parser = argparse.ArgumentParser(description="collecten test runner")
	parser.add_argument("--serial-port", default=None)
	parser.add_argument("--num-channels", type=int, default=4)
	parser.add_argument("--no-eeg", action="store_true")
	parser.add_argument("--fullscreen", action="store_true")
	args = parser.parse_args()

	print("collecten: 7.5 Hz test", flush=True)
	print(f"duration={TEST_DURATION_S}s, target={TARGET_HZ} Hz", flush=True)

	stream = None
	fs: Optional[float] = None
	if not args.no_eeg:
		from acquisition.brainflow_stream import BrainFlowStream
		from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
		BoardShim.disable_board_logger()
		p = BrainFlowInputParams()
		if args.serial_port:
			p.serial_port = args.serial_port
		stream = BrainFlowStream(board_id=BoardIds.NEUROPAWN_KNIGHT_BOARD.value, params=p, num_channels=args.num_channels)
		stream.prepare_session()
		stream.start_stream()
		# configure input channels and references similar to collect_training_data.py
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
			raise RuntimeError(f"Board sampling rate is {fs:0.3f} Hz, expected 125.000 Hz")
		print(f"Board ready at {fs:0.3f} Hz", flush=True)

	# Run display + recording in parallel: start recording thread via helper
	if stream is not None:
		result: dict = {}
		def _rec_thread() -> None:
			try:
				freqs, mean_psd, eeg = record_and_compute_psd(stream, TEST_DURATION_S, DISCARD_S, fs)
				result['freqs'] = freqs
				result['mean_psd'] = mean_psd
				result['eeg'] = eeg
			except Exception as e:
				result['error'] = str(e)

		t = threading.Thread(target=_rec_thread, daemon=True)
		t.start()
		# Run visual flicker in main thread (pygame must run in main)
		visual_flicker(TEST_DURATION_S, TARGET_HZ, args.fullscreen)
		t.join(timeout=TEST_DURATION_S + 5)

		if 'error' in result:
			print(f"Recording error: {result['error']}", flush=True)
		elif 'freqs' in result:
			freqs = result['freqs']
			mean_psd = result['mean_psd']
			eeg = result.get('eeg')
			# print per-channel stats to debug zero-PSD
			if eeg is not None:
				print(f"Raw EEG shape={eeg.shape}", flush=True)
				for ch in range(eeg.shape[0]):
					ch_data = eeg[ch]
					print(f"  CH{ch}: min={float(ch_data.min()):0.6g}, max={float(ch_data.max()):0.6g}, mean={float(ch_data.mean()):0.6g}, std={float(ch_data.std()):0.6g}", flush=True)
				# save raw data
				try:
					from utils.config import SSVEPConfig
					config = SSVEPConfig(left_hz=TARGET_HZ, right_hz=TARGET_HZ)
					config.data_raw_dir.mkdir(parents=True, exist_ok=True)
					ts = time.strftime("%Y%m%d_%H%M%S")
					X = np.expand_dims(eeg, axis=0)  # (1, n_ch, n_samples)
					y = np.array([0], dtype=np.int64)
					x_path = config.data_raw_dir / f"X_{ts}.npy"
					y_path = config.data_raw_dir / f"y_{ts}.npy"
					np.save(x_path, X)
					np.save(y_path, y)
					print(f"Saved raw data -> {x_path}  {y_path}", flush=True)
				except Exception as e:
					print(f"Failed to save raw data: {e}", flush=True)
			mask = (freqs >= TARGET_HZ - 2.0) & (freqs <= TARGET_HZ + 2.0)
			peak_freq = float(freqs[mask][np.argmax(mean_psd[mask])]) if np.any(mask) else float(freqs[np.argmax(mean_psd)])
			peak_power = float(np.max(mean_psd[mask])) if np.any(mask) else float(np.max(mean_psd))
			idx = int(np.argmin(np.abs(freqs - TARGET_HZ)))
			print(f"Recorded EEG shape={mean_psd.shape} (mean over channels)", flush=True)
			print(f"Peak freq in target band={peak_freq:0.3f} Hz (power={peak_power:0.3g})", flush=True)
			print(f"Power at {TARGET_HZ:0.3f} Hz -> bin {freqs[idx]:0.3f} Hz, power={mean_psd[idx]:0.3g}", flush=True)
		else:
			print("Recording did not complete in time.", flush=True)
		try:
			stream.stop_stream()
		except Exception:
			pass
		stream.release_session()
	else:
		# no-eeg mode: visual only
		visual_flicker(TEST_DURATION_S, TARGET_HZ, args.fullscreen)
		print("--no-eeg: visual test completed.", flush=True)


if __name__ == "__main__":
	main()
