import csv
import numpy as np
import os
import argparse
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--metric", type=str, default="reward")
    parser.add_argument("--smooth", type=int, default=0)
    parser.add_argument("--out", type=str, default=os.path.join("training", "comparison_plot.png"))
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args()

def _read_series(log_path, metric):
    if not os.path.exists(log_path):
        print(f"Warning: Log file {log_path} not found. Returning empty list.")
        return []
    with open(log_path, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]
        if not rows:
            print(f"Warning: Invalid log file {log_path}. Returning empty list.")
            return []
        for row in rows:
            if row and row[0] == metric:
                return [float(x) for x in row[1:] if x]
        if metric == "reward" and len(rows) >= 3 and rows[2] and rows[2][0] == "reward":
            return [float(x) for x in rows[2][1:] if x]
        print(f"Warning: Metric {metric} not found in {log_path}. Returning empty list.")
        return []

def _smooth(x, window):
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(np.asarray(x, dtype=np.float64), kernel, mode="valid").tolist()

def main():
    args = _parse_args()
    labels = args.labels if args.labels else [os.path.basename(p) for p in args.logs]
    if len(labels) != len(args.logs):
        labels = [os.path.basename(p) for p in args.logs]

    series_list = []
    for p in args.logs:
        s = _read_series(p, args.metric)
        series_list.append(s)

    if not any(series_list):
        print("No data available to plot.")
        return

    plt.figure(figsize=(10, 6))
    for label, s in zip(labels, series_list):
        if not s:
            continue
        y = _smooth(s, args.smooth) if args.smooth else s
        x = range(1, len(y) + 1)
        plt.plot(x, y, label=label)

    plt.xlabel('Episode')
    plt.ylabel(args.metric)
    title = args.title or f"Comparison: {args.metric}"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
