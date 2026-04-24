import os
import statistics
from typing import Dict, List

from benchmark_focus import TefnetLoader, make_sample_plans, run_policy


def summarize(results: List[Dict], key: str):
    vals = [r[key] for r in results]
    return statistics.mean(vals), statistics.pstdev(vals)


def main():
    seeds = [42, 52, 62]
    train_episodes = 120
    eval_episodes = 30
    sample_size = 15

    methods = {
        "QL": "rl",
        "DQN_Full": "dqn",
        "DQN_NoReplayTarget": "dqn_no_rt",
        "DQN_NoGuidedExplore": "dqn_no_guide",
    }
    keys = [
        "risk_repair_rate",
        "risk_after",
        "low_qot_after",
        "avg_osnr_after",
        "avg_reroute_actions",
        "control_messages",
    ]

    loader = TefnetLoader("data/tefnet_nodes.csv", "data/tefnet_links.csv", "data/tefnet_traffic.csv")
    topo = loader.load_topology()
    demands = loader.load_traffic_demands()

    all_res = {name: [] for name in methods}
    for seed in seeds:
        train_plans = make_sample_plans(len(demands), train_episodes, sample_size, seed)
        eval_plans = make_sample_plans(len(demands), eval_episodes, sample_size, seed + 1)
        for name, policy in methods.items():
            all_res[name].append(run_policy(policy, topo, demands, train_plans, eval_plans, seed=seed))

    print("=== DQN 消融实验（多种子均值±标准差）===")
    for key in keys:
        line = [f"{key:>22s}"]
        for name in methods:
            mean, std = summarize(all_res[name], key)
            line.append(f"{name:>20s}={mean:7.4f}±{std:6.4f}")
        print(" | ".join(line))

    for temp_path in (
        "data/q_table_memory_benchmark_tmp.json",
        "data/dqn_policy_benchmark_tmp.pt",
        "data/dqn_policy_no_rt_benchmark_tmp.pt",
        "data/dqn_policy_no_guide_benchmark_tmp.pt",
        "data/vdn_lite_policy_benchmark_tmp.pt",
    ):
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    main()
