import json
import os

os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_metric(data, metric_key: str, title: str, filename: str):
    levels = list(data["load_levels"].keys())
    methods = data["methods"]

    x = range(len(levels))
    plt.figure(figsize=(8, 5))
    for method in methods:
        y = [data["results"][lvl][method][metric_key]["mean"] for lvl in levels]
        plt.plot(x, y, marker="o", label=method.upper())

    plt.xticks(list(x), [lvl.upper() for lvl in levels])
    plt.xlabel("Load Level")
    plt.ylabel(metric_key)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_dir = "output_images"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"已生成图表: {out_path}")


def main():
    data = load_data("docs/load_level_results.json")
    draw_metric(data, "risk_repair_rate", "Risk Repair Rate Across Load Levels", "load_risk_repair_rate.png")
    draw_metric(data, "avg_osnr_after", "Average OSNR Across Load Levels", "load_avg_osnr.png")
    draw_metric(data, "control_messages", "Control Messages Across Load Levels", "load_control_messages.png")


if __name__ == "__main__":
    main()
