"""CLI demo: load a trained SSVEP model and print LEFT / RIGHT predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brainflow.board_shim import BoardShim  # noqa: E402
from realtime.inference import RealtimeSSVEPDecoder  # noqa: E402


def main() -> None:
    BoardShim.disable_board_logger()
    parser = argparse.ArgumentParser(description="SSVEP real-time demo (BrainFlow + sklearn).")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "data" / "processed" / "models" / "ssvep_pipeline.joblib",
        help="Path to joblib pipeline saved from the training notebook.",
    )
    parser.add_argument(
        "--max-predictions",
        type=int,
        default=30,
        help="Stop after this many windows (default: 30).",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        print(
            f"Model not found at {args.model}.\n"
            "Train and save a pipeline from notebooks/04_model_training.ipynb first."
        )
        sys.exit(1)

    decoder = RealtimeSSVEPDecoder(model_path=args.model)
    print("Streaming (synthetic board by default). Press Ctrl+C to stop.\n")
    count = 0
    try:
        for side, cls in decoder.run_forever():
            count += 1
            print(f"[{count:03d}] prediction={side} (class={cls})")
            if count >= args.max_predictions:
                break
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
