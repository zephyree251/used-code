import contextlib
import copy
import io
import os
import random
import statistics
from typing import Dict, List

import numpy as np

from agents.action import ActionAgent
from agents.decision_rl import DQNDecisionAgent, RLDecisionAgent, RuleDecisionAgent, VDNLiteDecisionAgent, default_q_table
from agents.evaluation import EvaluationAgent
from agents.evidence import EvidenceAgent
from agents.intent import IntentAgent
from agents.perception import PerceptionAgent
from core.context import SystemContext
from utils.decision_features import build_link_load_map, compute_path_length
from utils.tefnet_loader import TefnetLoader


def make_sample_plans(total_demands: int, episodes: int, sample_size: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    all_indices = list(range(total_demands))
    return [rng.sample(all_indices, sample_size) for _ in range(episodes)]


def _count_risks(risks: List[Dict]) -> Dict[str, int]:
    out = {"all": len(risks), "congestion": 0, "low_qot": 0}
    for item in risks:
        t = item.get("type", "")
        if t == "High_Congestion":
            out["congestion"] += 1
        elif t == "Low_QoT":
            out["low_qot"] += 1
    return out


def _avg_osnr(services: List[Dict]) -> float:
    vals = [float(s.get("osnr", 0.0)) for s in services if float(s.get("osnr", 0.0)) > 0]
    if not vals:
        return 0.0
    return float(statistics.mean(vals))


def _action_stats(logs: List[str]) -> Dict[str, int]:
    out = {"reroute": 0, "power_boost": 0, "maintain": 0}
    for log in logs:
        if "Reroute" in log:
            out["reroute"] += 1
        elif "Power_Boost" in log:
            out["power_boost"] += 1
        elif "Maintain" in log:
            out["maintain"] += 1
    return out


def _avg_path_length(graph, services: List[Dict]) -> float:
    vals = []
    for srv in services:
        path = srv.get("path", [])
        if len(path) >= 2:
            vals.append(compute_path_length(graph, path))
    return float(statistics.mean(vals)) if vals else 0.0


def _max_network_utilization(services: List[Dict]) -> float:
    loads = build_link_load_map(services)
    if not loads:
        return 0.0
    return max(load / 8000.0 for load in loads.values())


def _control_message_count(context) -> int:
    reroute_msgs = sum(1 for log in context.action_logs if "Reroute" in log)
    return len(context.intents) + len(context.decisions) + reroute_msgs


def _run_episode(idx_list, graph, all_demands, perception, intent, evidence, decision, action, evaluation, learn: bool):
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
        act = _action_stats(ctx.action_logs)

    return ctx, risk_before, risk_after, act


def run_policy(
    policy: str,
    graph,
    all_demands: List[Dict],
    train_plans: List[List[int]],
    eval_plans: List[List[int]],
    seed: int,
) -> Dict[str, float]:
    random.seed(seed)
    np.random.seed(seed)

    perception = PerceptionAgent("Perception")
    intent = IntentAgent("Intent")
    evidence = EvidenceAgent("Evidence")
    action = ActionAgent("Action")

    if policy == "rl":
        decision = RLDecisionAgent(
            "Decision(Q-Learning)",
            memory_file="data/q_table_memory_benchmark_tmp.json",
            load_existing=False,
        )
        decision.q_table = default_q_table()
        evaluation = EvaluationAgent("Evaluation", decision)
    elif policy == "dqn":
        decision = DQNDecisionAgent(
            "Decision(DQN)",
            memory_file="data/dqn_policy_benchmark_tmp.pt",
            load_existing=False,
        )
        if os.path.exists(decision.memory_file):
            os.remove(decision.memory_file)
        evaluation = EvaluationAgent("Evaluation", decision)
    elif policy == "dqn_3layer":
        decision = DQNDecisionAgent(
            "Decision(DQN-3Layer)",
            memory_file="data/dqn_policy_3layer_benchmark_tmp.pt",
            load_existing=False,
            hidden_dims=(64, 32, 16),
        )
        if os.path.exists(decision.memory_file):
            os.remove(decision.memory_file)
        evaluation = EvaluationAgent("Evaluation", decision)
    elif policy == "dqn_no_rt":
        decision = DQNDecisionAgent(
            "Decision(DQN-NoReplayTarget)",
            memory_file="data/dqn_policy_no_rt_benchmark_tmp.pt",
            load_existing=False,
            use_replay_target=False,
        )
        if os.path.exists(decision.memory_file):
            os.remove(decision.memory_file)
        evaluation = EvaluationAgent("Evaluation", decision)
    elif policy == "dqn_no_guide":
        decision = DQNDecisionAgent(
            "Decision(DQN-NoGuidedExplore)",
            memory_file="data/dqn_policy_no_guide_benchmark_tmp.pt",
            load_existing=False,
            use_guided_exploration=False,
        )
        if os.path.exists(decision.memory_file):
            os.remove(decision.memory_file)
        evaluation = EvaluationAgent("Evaluation", decision)
    elif policy == "vdn":
        decision = VDNLiteDecisionAgent(
            "Decision(VDN-lite)",
            memory_file="data/vdn_lite_policy_benchmark_tmp.pt",
            load_existing=False,
        )
        if os.path.exists(decision.memory_file):
            os.remove(decision.memory_file)
        evaluation = EvaluationAgent("Evaluation", decision)
    else:
        decision = RuleDecisionAgent("Decision(Rule)")
        evaluation = None

    if policy in ("rl", "dqn", "vdn", "dqn_no_rt", "dqn_no_guide", "dqn_3layer"):
        for idx_list in train_plans:
            _run_episode(idx_list, graph, all_demands, perception, intent, evidence, decision, action, evaluation, learn=True)
        if hasattr(decision, "set_eval_mode"):
            decision.set_eval_mode()
        elif hasattr(decision, "epsilon"):
            decision.epsilon = 0.0

    before_total = []
    after_total = []
    before_congestion = []
    after_congestion = []
    before_low_qot = []
    after_low_qot = []
    osnr_after = []
    reroute_count = []
    boost_count = []
    avg_path_length = []
    max_link_util = []
    control_messages = []

    for idx_list in eval_plans:
        ctx, risk_before, risk_after, act = _run_episode(
            idx_list, graph, all_demands, perception, intent, evidence, decision, action, evaluation, learn=False
        )

        before_total.append(risk_before["all"])
        after_total.append(risk_after["all"])
        before_congestion.append(risk_before["congestion"])
        after_congestion.append(risk_after["congestion"])
        before_low_qot.append(risk_before["low_qot"])
        after_low_qot.append(risk_after["low_qot"])
        osnr_after.append(_avg_osnr(ctx.active_services))
        reroute_count.append(act["reroute"])
        boost_count.append(act["power_boost"])
        avg_path_length.append(_avg_path_length(graph, ctx.active_services))
        max_link_util.append(_max_network_utilization(ctx.active_services))
        control_messages.append(_control_message_count(ctx))

    def mean(vals: List[float]) -> float:
        return float(statistics.mean(vals)) if vals else 0.0

    before = mean(before_total)
    after = mean(after_total)
    repair_rate = 0.0 if before <= 1e-9 else (before - after) / before

    return {
        "episodes": float(len(eval_plans)),
        "risk_before": before,
        "risk_after": after,
        "risk_repair_rate": repair_rate,
        "congestion_before": mean(before_congestion),
        "congestion_after": mean(after_congestion),
        "low_qot_before": mean(before_low_qot),
        "low_qot_after": mean(after_low_qot),
        "avg_osnr_after": mean(osnr_after),
        "avg_reroute_actions": mean(reroute_count),
        "avg_boost_actions": mean(boost_count),
        "avg_path_length": mean(avg_path_length),
        "max_link_utilization": mean(max_link_util),
        "control_messages": mean(control_messages),
    }


def print_compare(metrics_map: Dict[str, Dict[str, float]]):
    print("\n=== 研究聚焦实验：Rule vs Q-Learning vs DQN vs VDN-lite ===")
    keys = [
        "risk_before",
        "risk_after",
        "risk_repair_rate",
        "congestion_before",
        "congestion_after",
        "low_qot_before",
        "low_qot_after",
        "avg_osnr_after",
        "avg_reroute_actions",
        "avg_boost_actions",
        "avg_path_length",
        "max_link_utilization",
        "control_messages",
    ]
    for k in keys:
        print(
            f"{k:>20s} | "
            f"Rule={metrics_map['rule'][k]:8.4f} | "
            f"QL={metrics_map['rl'][k]:8.4f} | "
            f"DQN={metrics_map['dqn'][k]:8.4f} | "
            f"VDN={metrics_map['vdn'][k]:8.4f}"
        )


if __name__ == "__main__":
    TRAIN_EPISODES = 120
    EVAL_EPISODES = 30
    SAMPLE_SIZE = 15
    SEED = 42

    loader = TefnetLoader("data/tefnet_nodes.csv", "data/tefnet_links.csv", "data/tefnet_traffic.csv")
    topo = loader.load_topology()
    demands = loader.load_traffic_demands()
    train_plans = make_sample_plans(len(demands), TRAIN_EPISODES, SAMPLE_SIZE, SEED)
    eval_plans = make_sample_plans(len(demands), EVAL_EPISODES, SAMPLE_SIZE, SEED + 1)

    metrics = {
        "rule": run_policy("rule", topo, demands, train_plans, eval_plans, seed=SEED),
        "rl": run_policy("rl", topo, demands, train_plans, eval_plans, seed=SEED),
        "dqn": run_policy("dqn", topo, demands, train_plans, eval_plans, seed=SEED),
        "vdn": run_policy("vdn", topo, demands, train_plans, eval_plans, seed=SEED),
    }
    print_compare(metrics)

    for temp_path in (
        "data/q_table_memory_benchmark_tmp.json",
        "data/dqn_policy_benchmark_tmp.pt",
        "data/dqn_policy_3layer_benchmark_tmp.pt",
        "data/dqn_policy_no_rt_benchmark_tmp.pt",
        "data/dqn_policy_no_guide_benchmark_tmp.pt",
        "data/vdn_lite_policy_benchmark_tmp.pt",
    ):
        if os.path.exists(temp_path):
            os.remove(temp_path)
