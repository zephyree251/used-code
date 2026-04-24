import contextlib
import copy
import csv
import io
import os
import random
import statistics
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.action import ActionAgent
from agents.decision_rl import DQNDecisionAgent
from agents.evaluation import EvaluationAgent
from agents.evidence import EvidenceAgent
from agents.intent import IntentAgent
from agents.perception import PerceptionAgent
from benchmark_focus import _avg_osnr, _count_risks
from core.context import SystemContext
from utils.tefnet_loader import TefnetLoader


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
        "risk_repair_rate": float(repair),
        "risk_after": float(after_all),
        "low_qot_after": float(risk_after["low_qot"]),
        "avg_osnr_after": float(_avg_osnr(ctx.active_services)),
    }


def evaluate_model(graph, all_demands, decision, sample_size=15, eval_episodes=30, eval_seed=2026):
    perception = PerceptionAgent("Perception")
    intent = IntentAgent("Intent")
    evidence = EvidenceAgent("Evidence")
    action = ActionAgent("Action")

    rng = random.Random(eval_seed)
    all_idx = list(range(len(all_demands)))

    old_eps = decision.epsilon
    decision.set_eval_mode()
    metrics = []
    for _ in range(eval_episodes):
        idx_list = rng.sample(all_idx, sample_size)
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


def plot_single_metric(x, y, title, ylabel, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=1.8)
    plt.xlabel("Training Episodes")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    sample_size = 15
    max_train_episodes = 200
    eval_interval = 5
    eval_episodes = 30
    train_seed = 2025
    eval_seed = 2026

    out_dir = "output_images/dqn_dense_curves"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = "docs/dqn_training_curve_dense.csv"

    loader = TefnetLoader("data/tefnet_nodes.csv", "data/tefnet_links.csv", "data/tefnet_traffic.csv")
    graph = loader.load_topology()
    all_demands = loader.load_traffic_demands()

    decision = DQNDecisionAgent("Decision(DQN-Dense)", load_existing=True)
    evaluation = EvaluationAgent("Evaluation", decision)
    perception = PerceptionAgent("Perception")
    intent = IntentAgent("Intent")
    evidence = EvidenceAgent("Evidence")
    action = ActionAgent("Action")

    rng = random.Random(train_seed)
    all_idx = list(range(len(all_demands)))
    curve = []

    for episode in range(1, max_train_episodes + 1):
        idx_list = rng.sample(all_idx, sample_size)
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

        if episode % eval_interval == 0:
            metrics = evaluate_model(
                graph,
                all_demands,
                decision,
                sample_size=sample_size,
                eval_episodes=eval_episodes,
                eval_seed=eval_seed,
            )
            curve.append({"episode": episode, **metrics})
            print(
                f"[Eval@{episode}] repair={metrics['risk_repair_rate']:.4f}, "
                f"risk_after={metrics['risk_after']:.4f}, "
                f"low_qot={metrics['low_qot_after']:.4f}, osnr={metrics['avg_osnr_after']:.4f}"
            )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode", "risk_repair_rate", "risk_after", "low_qot_after", "avg_osnr_after"],
        )
        writer.writeheader()
        writer.writerows(curve)

    x = [c["episode"] for c in curve]
    plot_single_metric(
        x,
        [c["risk_repair_rate"] for c in curve],
        "DQN Dense Curve - Risk Repair Rate",
        "risk_repair_rate",
        os.path.join(out_dir, "dqn_risk_repair_rate_dense.png"),
    )
    plot_single_metric(
        x,
        [c["risk_after"] for c in curve],
        "DQN Dense Curve - Risk After",
        "risk_after",
        os.path.join(out_dir, "dqn_risk_after_dense.png"),
    )
    plot_single_metric(
        x,
        [c["low_qot_after"] for c in curve],
        "DQN Dense Curve - Low QoT After",
        "low_qot_after",
        os.path.join(out_dir, "dqn_low_qot_after_dense.png"),
    )
    plot_single_metric(
        x,
        [c["avg_osnr_after"] for c in curve],
        "DQN Dense Curve - Avg OSNR After",
        "avg_osnr_after",
        os.path.join(out_dir, "dqn_avg_osnr_after_dense.png"),
    )

    print("\n=== 更密曲线已生成 ===")
    print(f"CSV: {csv_path}")
    print(f"图目录: {out_dir}")


if __name__ == "__main__":
    main()
