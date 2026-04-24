import os
import json
import statistics
from typing import Dict, List

from benchmark_focus import TefnetLoader, make_sample_plans, run_policy


def aggregate_metrics(results: List[Dict], metric_keys: List[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for key in metric_keys:
        values = [item[key] for item in results]
        out[key] = {
            "mean": float(statistics.mean(values)),
            "std": float(statistics.pstdev(values)),
        }
    return out


def print_table(level_name: str, methods: List[str], aggregated: Dict[str, Dict[str, Dict[str, float]]], metric_keys: List[str]):
    print(f"\n=== 负载档位: {level_name} ===")
    for metric in metric_keys:
        parts = [f"{metric:>20s}"]
        for method in methods:
            mean = aggregated[method][metric]["mean"]
            std = aggregated[method][metric]["std"]
            parts.append(f"{method.upper():>4s}={mean:7.4f}±{std:6.4f}")
        print(" | ".join(parts))


def main():
    seeds = [42, 52, 62]
    train_episodes = 120
    eval_episodes = 30

    include_vdn = os.environ.get("INCLUDE_VDN", "0") == "1"
    methods = ["rule", "rl", "dqn"] + (["vdn"] if include_vdn else [])

    # 每轮活跃业务数量，代表不同负载强度
    load_levels = {
        "low": 10,
        "mid": 15,
        "high": 25,
    }

    metric_keys = [
        "risk_repair_rate",
        "risk_after",
        "congestion_after",
        "low_qot_after",
        "avg_osnr_after",
        "avg_reroute_actions",
        "avg_boost_actions",
        "control_messages",
    ]

    loader = TefnetLoader("data/tefnet_nodes.csv", "data/tefnet_links.csv", "data/tefnet_traffic.csv")
    topo = loader.load_topology()
    demands = loader.load_traffic_demands()

    all_level_results = {}

    for level_name, sample_size in load_levels.items():
        level_results = {method: [] for method in methods}
        for seed in seeds:
            train_plans = make_sample_plans(len(demands), train_episodes, sample_size, seed)
            eval_plans = make_sample_plans(len(demands), eval_episodes, sample_size, seed + 1)
            for method in methods:
                metrics = run_policy(method, topo, demands, train_plans, eval_plans, seed=seed)
                level_results[method].append(metrics)

        aggregated = {
            method: aggregate_metrics(level_results[method], metric_keys) for method in methods
        }
        all_level_results[level_name] = aggregated
        print_table(level_name, methods, aggregated, metric_keys)

    # 保存机器可读结果，便于绘图和论文复用
    output_path = "docs/load_level_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seeds": seeds,
                "methods": methods,
                "load_levels": load_levels,
                "metrics": metric_keys,
                "results": all_level_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n已保存负载分层结果: {output_path}")

    # 清理 benchmark 临时模型
    for temp_path in (
        "data/q_table_memory_benchmark_tmp.json",
        "data/dqn_policy_benchmark_tmp.pt",
        "data/vdn_lite_policy_benchmark_tmp.pt",
    ):
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("\n=== 负载分层实验完成 ===")


if __name__ == "__main__":
    main()
