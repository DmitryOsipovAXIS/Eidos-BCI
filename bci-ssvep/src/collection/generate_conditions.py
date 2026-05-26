import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=20,
                    help='Trials per class (default 20)')
args = parser.parse_args()

rows = [{"label": v, "side": "LEFT" if v == 0 else "RIGHT"}
        for _ in range(args.trials)
        for v in [0, 1]]

with open("src/collection/conditions.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["label", "side"])
    writer.writeheader()
    writer.writerows(rows)

print(
    f"Generated {len(rows)} trials ({args.trials} per class) -> conditions.csv")
