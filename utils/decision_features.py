import itertools
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


OSNR_THRESHOLD_DB = 30.0
LINK_CAPACITY = 8000.0


def clean_service_id(raw_id) -> str:
    return str(raw_id).strip().replace("RISK-", "").replace("INT-", "")


def find_service(context, service_id) -> Optional[Dict]:
    target_id = clean_service_id(service_id)
    return next((s for s in context.active_services if clean_service_id(s.get("id")) == target_id), None)


def build_link_load_map(services: List[Dict], skip_service_id: Optional[str] = None) -> Dict[Tuple[str, str], float]:
    loads: Dict[Tuple[str, str], float] = {}
    skip_id = clean_service_id(skip_service_id) if skip_service_id else None

    for srv in services:
        if skip_id and clean_service_id(srv.get("id")) == skip_id:
            continue

        path = srv.get("path", [])
        bandwidth = float(srv.get("bandwidth", 0.0))
        for idx in range(len(path) - 1):
            edge = (path[idx], path[idx + 1])
            loads[edge] = loads.get(edge, 0.0) + bandwidth

    return loads


def compute_path_length(graph, path: List[str]) -> float:
    total = 0.0
    for idx in range(len(path) - 1):
        edge_data = graph.get_edge_data(path[idx], path[idx + 1], default={})
        total += float(edge_data.get("length_km", 50.0))
    return total


def compute_max_link_utilization(context, service: Dict, include_self: bool = True) -> float:
    skip_service_id = None if include_self else service.get("id")
    loads = build_link_load_map(context.active_services, skip_service_id=skip_service_id)

    if include_self:
        path = service.get("path", [])
        bandwidth = float(service.get("bandwidth", 0.0))
        for idx in range(len(path) - 1):
            edge = (path[idx], path[idx + 1])
            loads[edge] = loads.get(edge, 0.0) + bandwidth

    max_util = 0.0
    for idx in range(len(service.get("path", [])) - 1):
        edge = (service["path"][idx], service["path"][idx + 1])
        util = loads.get(edge, 0.0) / LINK_CAPACITY
        max_util = max(max_util, util)
    return max_util


def count_alternative_paths(context, service: Dict, k: int = 3) -> int:
    if not service.get("source") or not service.get("target"):
        return 0

    current_path = service.get("path", [])
    req_bw = float(service.get("bandwidth", 0.0))
    current_usage = build_link_load_map(context.active_services, skip_service_id=service.get("id"))

    def filter_edge(u, v):
        remaining = LINK_CAPACITY - current_usage.get((u, v), 0.0)
        return remaining >= req_bw

    safe_graph = nx.subgraph_view(context.graph, filter_edge=filter_edge)

    try:
        generator = nx.shortest_simple_paths(
            safe_graph,
            source=service["source"],
            target=service["target"],
            weight="length_km",
        )
        k_paths = list(itertools.islice(generator, k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return 0
    except Exception:
        return 0

    return sum(1 for path in k_paths if path != current_path)


def encode_state(context, service: Dict, intent: Dict) -> np.ndarray:
    path = service.get("path", [])
    path_length = compute_path_length(context.graph, path)
    hop_count = max(0, len(path) - 1)
    osnr = float(service.get("osnr", 0.0))
    bandwidth = float(service.get("bandwidth", 0.0))
    max_util = compute_max_link_utilization(context, service, include_self=True)
    alt_path_count = count_alternative_paths(context, service, k=3)

    service_class = 1.0 if "Core_VIP" in intent.get("user_level", "") else 0.0
    risk_type = 1.0 if intent.get("issue_type") == "Congestion" else 0.0
    bandwidth_norm = min(bandwidth / LINK_CAPACITY, 1.5)
    path_len_norm = min(path_length / 2000.0, 2.0)
    hop_count_norm = min(hop_count / 15.0, 1.5)
    osnr_margin_norm = np.clip((osnr - OSNR_THRESHOLD_DB) / 10.0, -2.0, 2.0)
    max_link_util_norm = np.clip(max_util, 0.0, 2.0)
    alt_path_count_norm = min(alt_path_count / 3.0, 1.0)

    return np.array(
        [
            service_class,
            risk_type,
            bandwidth_norm,
            path_len_norm,
            hop_count_norm,
            osnr_margin_norm,
            max_link_util_norm,
            alt_path_count_norm,
        ],
        dtype=np.float32,
    )


def snapshot_service_metrics(context, service: Dict) -> Dict:
    path = service.get("path", [])
    return {
        "osnr": float(service.get("osnr", 0.0)),
        "max_util": float(compute_max_link_utilization(context, service, include_self=True)),
        "path_length": float(compute_path_length(context.graph, path)),
        "hop_count": float(max(0, len(path) - 1)),
        "path": list(path),
        "bandwidth": float(service.get("bandwidth", 0.0)),
    }
