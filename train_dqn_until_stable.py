import contextlib
import copy
import csv
import io
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statistics

from agents.action import ActionAgent
from agents.decision_rl import DQNDecisionAgent
from agents.evaluation import EvaluationAgent
from agents.evidence import EvidenceAgent
from agents.intent import IntentAgent
from agents.perception import PerceptionAgent
from benchmark_focus import _avg_osnr, _count_risks
from core.context import SystemContext
from utils.tefnet_loader import TefnetLoader


@dataclass
class TrainConfig:
    sample_size: int = 15
    max_train_episodes: int = 500
    eval_interval: int = 20
    eval_episodes: int = 30
    eval_seed: int = 2026
    min_delta: float = 0.005
    patience_evals: int = 4
    reset_model: bool = False


def run_episode(
    graph,
    all_demands: List[Dict],
    idx_list: List[int],
    perception,
    intent,
    evidence,
    decision,
    action,
    evaluation=None,
    learn: bool = False,
):
    ctx = SystemContext()
    ctx.graph = graph
    ctx.all_demands = all_demands
    ctx.active_services = [copy.deepcopy(all_demands[i]) for i in idx_list]
    ctx.risk_report = []
    ctx.intents = []
    ctx.evidence = []
    ctx.decisions = []
    ctx.action_logs = []

    with contextlib.redirect_stdout(io.StringIO()):
        perception.process(ctx)
        risk_before = _count_risks(ctx.risk_report)

        intent.process(ctx)
        evidence.process(ctx)
        decision.process(ctx)
        action.process(ctx)
        if learn and evaluation is not None:
            evaluation.process(ctx)

        perception.process(ctx)
        risk_after = _count_risks(ctx.risk_report)

    before_all = risk_before["all"]
    after_all = risk_after["all"]
    repair = 0.0 if before_all <= 1e-9 else (before_all - after_all) / before_all
    return {
        "risk_before": float(before_all),
        "risk_after": float(after_all),
        "risk_repair_rate": float(repair),
        "low_qot_after": float(risk_after["low_qot"]),
        "avg_osnr_after": float(_avg_osnr(ctx.active_services)),
    }


def evaluate_model(graph, all_demands, decision, config: TrainConfig):
    perception = PerceptionAgent("Perception")
    intent = IntentAgent("Intent")
    evidence = EvidenceAgent("Evidence")
    action = ActionAgent("Action")

    rng = random.Random(config.eval_seed)
    all_idx = list(range(len(all_demands)))

    old_eps = decision.epsilon
    decision.set_eval_mode()
    metrics = []
    for _ in range(config.eval_episodes):
        idx_list = rng.sample(all_idx, config.sample_size)
        metrics.append(
            run_episode(
                graph,
                all_demands,
                idx_list,
                perception,
                intent,
                evidence,
                decision,
                action,
                evaluation=None,
                learn=False,
            )
        )
    decision.epsilon = old_eps

    return {
        "risk_repair_rate": statistics.mean([m["risk_repair_rate"] for m in metrics]),
        "risk_after": statistics.mean([m["risk_after"] for m in metrics]),
        "low_qot_after": statistics.mean([m["low_qot_after"] for m in metrics]),
        "avg_osnr_after": statistics.mean([m["avg_osnr_after"] for m in metrics]),
    }


def save_curve(curve: List[Dict], csv_path: str, img_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode", "risk_repair_rate", "risk_after", "low_qot_after", "avg_osnr_after"],
        )
        writer.writeheader()
        writer.writerows(curve)

    x = [c["episode"] for c in curve]
    y1 = [c["risk_repair_rate"] for c in curve]
    y2 = [c["low_qot_after"] for c in curve]
    y3 = [c["avg_osnr_after"] for c in curve]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, marker="o", label="risk_repair_rate")
    plt.plot(x, y2, marker="s", label="low_qot_after")
    plt.plot(x, y3, marker="^", label="avg_osnr_after")
    plt.xlabel("Training Episodes")
    plt.ylabel("Metric Value")
    plt.title("DQN Training Progress Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path, dpi=220)
    plt.close()


def main():
    config = TrainConfig()

    loader = TefnetLoader("data/tefnet_nodes.csv", "data/tefnet_links.csv", "data/tefnet_traffic.csv")
    graph = loader.load_topology()
    all_demands = loader.load_traffic_demands()

    if config.reset_model and os.path.exists("data/dqn_policy_memory.pt"):
        os.remove("data/dqn_policy_memory.pt")

    decision = DQNDecisionAgent("Decision(DQN)", load_existing=True)
    evaluation = EvaluationAgent("Evaluation", decision)
    perception = PerceptionAgent("Perception")
    intent = IntentAgent("Intent")
    evidence = EvidenceAgent("Evidence")
    action = ActionAgent("Action")

    rng = random.Random(2025)
    all_idx = list(range(len(all_demands)))

    curve = []
    best_score = -1e9
    no_improve = 0
    stop_episode = config.max_train_episodes

    for episode in range(1, config.max_train_episodes + 1):
        idx_list = rng.sample(all_idx, config.sample_size)
        run_episode(
            graph,
            all_demands,
            idx_list,
            perception,
            intent,
            evidence,
            decision,
            action,
            evaluation=evaluation,
            learn=True,
        )

        if episode % config.eval_interval == 0:
            metrics = evaluate_model(graph, all_demands, decision, config)
            row = {"episode": episode, **metrics}
            curve.append(row)
            print(
                f"[Eval@{episode}] repair={metrics['risk_repair_rate']:.4f}, "
                f"low_qot={metrics['low_qot_after']:.4f}, osnr={metrics['avg_osnr_after']:.4f}"
            )

            score = metrics["risk_repair_rate"]
            if score > best_score + config.min_delta:
                best_score = score
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= config.patience_evals:
                    stop_episode = episode
                    print(f"检测到收益趋稳，提前停止于 episode={episode}")
                    break

    csv_path = "docs/dqn_training_curve.csv"
    img_path = "output_images/dqn_training_curve.png"
    save_curve(curve, csv_path, img_path)

    print("\n=== 训练完成 ===")
    print(f"停止轮数: {stop_episode}")
    print(f"最佳修复率: {best_score:.4f}")
    print(f"曲线数据: {csv_path}")
    print(f"曲线图: {img_path}")


if __name__ == "__main__":
    main()
