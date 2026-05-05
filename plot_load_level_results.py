import json
import os

os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def setup_chinese_font():
    candidates = [
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            font_name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return


def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_metric(data, metric_key: str, title: str, filename: str):
    levels = list(data["load_levels"].keys())
    methods = data["methods"]
    label_map = {
        "rule": "Rule",
        "rl": "Q-Learning",
        "dqn": "DQN",
    }
    level_map = {
        "low": "低负载",
        "mid": "中负载",
        "medium": "中负载",
        "high": "高负载",
    }
    ylabel_map = {
        "risk_repair_rate": "风险修复率",
        "avg_osnr_after": "平均OSNR",
        "control_messages": "控制消息数量",
    }

    x = range(len(levels))
    plt.figure(figsize=(8, 5))
    for method in methods:
        y = [data["results"][lvl][method][metric_key]["mean"] for lvl in levels]
        plt.plot(x, y, marker="o", label=label_map.get(method, method.upper()))

    plt.xticks(list(x), [level_map.get(lvl, lvl) for lvl in levels])
    plt.xlabel("负载场景")
    plt.ylabel(ylabel_map.get(metric_key, metric_key))
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
    setup_chinese_font()
    data = load_data("docs/load_level_results.json")
    draw_metric(data, "risk_repair_rate", "不同负载下风险修复率对比", "load_risk_repair_rate.png")
    draw_metric(data, "avg_osnr_after", "不同负载下平均OSNR对比", "load_avg_osnr.png")
    draw_metric(data, "control_messages", "不同负载下控制消息数量对比", "load_control_messages.png")


if __name__ == "__main__":
    main()
