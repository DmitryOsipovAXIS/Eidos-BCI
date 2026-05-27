import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRIALS = 1
MODE = "both"
FREQ_LEFT = 7.5
FREQ_RIGHT = 12.0
SERIAL_PORT = "COM3"
NUM_CHANNELS = 8
NO_EEG = True

conditions_path = ROOT / "src" / "collection" / "conditions.csv"

if MODE == "both":
    rows = [{"side": side}
            for _ in range(TRIALS) for side in ["LEFT", "RIGHT"]]
elif MODE == "left":
    rows = [{"side": "LEFT"} for _ in range(TRIALS)]
else:
    rows = [{"side": "RIGHT"} for _ in range(TRIALS)]

with open(conditions_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["side"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} trials ({MODE}) -> {conditions_path}")

script = ROOT / "src" / "collection" / "collecting_lastrun.py"
cmd = [
    sys.executable, str(script),
    f"--freq-left={FREQ_LEFT}",
    f"--freq-right={FREQ_RIGHT}",
    f"--serial-port={SERIAL_PORT}",
    f"--num-channels={NUM_CHANNELS}",
]

if NO_EEG:
    cmd.append("--no-eeg")

print(f"Starting: {' '.join(cmd)}")
subprocess.run(cmd)
