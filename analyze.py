"""
Turns the CSV logs from pipeline.py into summary statistics and charts
for the performance whitepaper.

Usage:
    python analyze.py                      # analyses every results/metrics_*.csv
    python analyze.py results/metrics_X.csv
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd


def summarise(df, name):
    s = df["total_ms"]
    print(f"\n=== {name}  (mode: {df['mode'].iloc[0]}, frames: {len(df)}) ===")
    print(f"Mean latency   : {s.mean():.2f} ms")
    print(f"Median latency : {s.median():.2f} ms")
    print(f"P95 latency    : {s.quantile(0.95):.2f} ms")
    print(f"Max latency    : {s.max():.2f} ms")
    print(f"Std deviation  : {s.std():.2f} ms")
    print(f"Mean FPS       : {1000 / s.mean():.2f}")
    print(f"Avg objects/frame: {df['objects'].mean():.2f}")
    print(f"Mean stage times -> preprocess {df['preprocess_ms'].mean():.2f} | "
          f"threshold/post {df['threshold_or_post_ms'].mean():.2f} | "
          f"detect {df['detect_ms'].mean():.2f} ms")


def plot(df, name, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(f"Frame-by-frame metrics - {name}")

    axes[0, 0].plot(df["frame"], df["total_ms"], lw=0.8)
    axes[0, 0].set(title="Latency per frame", xlabel="frame", ylabel="ms")

    axes[0, 1].hist(df["total_ms"], bins=40)
    axes[0, 1].set(title="Latency distribution", xlabel="ms", ylabel="frames")

    axes[1, 0].plot(df["frame"], df["fps"], lw=0.8, color="tab:green")
    axes[1, 0].set(title="Rolling FPS", xlabel="frame", ylabel="FPS")

    axes[1, 1].plot(df["frame"], df["objects"], lw=0.8, color="tab:red")
    axes[1, 1].set(title="Objects detected per frame", xlabel="frame", ylabel="count")

    fig.tight_layout()
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Chart saved: {path}")


def main():
    files = sys.argv[1:] or sorted(glob.glob("results/metrics_*.csv"))
    if not files:
        raise SystemExit("No CSV files found. Run pipeline.py first.")
    outdir = os.path.join("results", "figures")
    os.makedirs(outdir, exist_ok=True)

    frames = []
    for f in files:
        df = pd.read_csv(f)
        name = os.path.splitext(os.path.basename(f))[0]
        summarise(df, name)
        plot(df, name, outdir)
        frames.append((name, df))

    if len(frames) > 1:                    # side-by-side comparison chart
        plt.figure(figsize=(8, 4))
        for name, df in frames:
            plt.plot(df["frame"], df["total_ms"], lw=0.8, label=f"{df['mode'].iloc[0]} ({name[-6:]})")
        plt.xlabel("frame"); plt.ylabel("latency (ms)"); plt.legend(); plt.title("Mode comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "comparison.png"), dpi=150)
        print("Chart saved: results/figures/comparison.png")


if __name__ == "__main__":
    main()
