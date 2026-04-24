import csv
import os

os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_average(values, window: int = 5):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot_metric(episodes, raw_values, smooth_values, title, ylabel, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, raw_values, color="#9aa0a6", linewidth=1.2, alpha=0.65, label="Raw")
    plt.plot(episodes, smooth_values, color="#1f77b4", linewidth=2.2, label="Smoothed (MA)")
    plt.xlabel("Training Episodes")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def main():
    csv_path = "docs/dqn_training_curve_dense.csv"
    out_dir = "output_images/dqn_dense_curves"
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    episodes = [int(r["episode"]) for r in rows]
    metrics = {
        "risk_repair_rate": "dqn_risk_repair_rate_dense_smoothed.png",
        "risk_after": "dqn_risk_after_dense_smoothed.png",
        "low_qot_after": "dqn_low_qot_after_dense_smoothed.png",
        "avg_osnr_after": "dqn_avg_osnr_after_dense_smoothed.png",
    }

    for metric, filename in metrics.items():
        raw_vals = [float(r[metric]) for r in rows]
        smooth_vals = moving_average(raw_vals, window=5)
        title = f"DQN Dense Curve - {metric} (Smoothed)"
        out_path = os.path.join(out_dir, filename)
        plot_metric(episodes, raw_vals, smooth_vals, title, metric, out_path)
        print(f"已生成平滑图: {out_path}")


if __name__ == "__main__":
    main()
