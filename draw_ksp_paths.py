#!/usr/bin/env python3
"""
仿照 output_images/reroute_*.png 的风格，绘制 K 条最短简单路径（KSP）。
与 agents/action.py 中一致：nx.shortest_simple_paths(..., weight='length_km')。

用法（在项目根目录）:
  python draw_ksp_paths.py
  python draw_ksp_paths.py --service INT-081
  python draw_ksp_paths.py --source N051 --target N052 --k 5

默认起讫点选为流量业务 INT-081（N051→N052）：相邻点间存在极短直连与多条长绕行，
K 条最短路在图上分离明显，比 N201→N203 等「路径重叠多」的 OD 更适合展示 KSP。
"""
import argparse
import itertools
import os
import sys

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx

# 保证可从项目根导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.tefnet_loader import TefnetLoader


def path_length_km(G, path):
    s = 0.0
    for i in range(len(path) - 1):
        d = G.get_edge_data(path[i], path[i + 1], default={})
        s += float(d.get("length_km", 50.0))
    return s


# 与「扫描流量 OD、选发散度高的业务」一致时的推荐默认（见 main 中说明）
DEFAULT_SRC = "N051"
DEFAULT_DST = "N052"

# 每条路径：线型轮换，避免仅靠颜色区分
_PATH_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


def draw_ksp_figure(G, source, target, k=5, output_dir="output_images"):
    os.makedirs(output_dir, exist_ok=True)
    pos = nx.kamada_kawai_layout(G)

    plt.figure(figsize=(14, 9))

    nx.draw_networkx_nodes(G, pos, node_size=28, node_color="#e8e8e8")
    nx.draw_networkx_edges(
        G, pos, edge_color="#d0d0d0", alpha=0.45, width=0.6, arrows=False
    )

    try:
        gen = nx.shortest_simple_paths(G, source, target, weight="length_km")
        paths = list(itertools.islice(gen, k))
    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
        print(f"无法求路径: {e}")
        return None

    if not paths:
        print("未找到任何路径")
        return None

    n_paths = len(paths)
    tab10 = plt.colormaps["tab10"]
    path_colors = [tab10(i / max(n_paths - 1, 1)) for i in range(n_paths)]

    # 先画较长路径，后画较短路径，使最短路径在最上层；线宽随名次略减，突出第 1 短路
    for rank in range(n_paths - 1, -1, -1):
        path = paths[rank]
        color = path_colors[rank]
        lw = 4.2 - rank * 0.55
        ls = _PATH_LINESTYLES[rank % len(_PATH_LINESTYLES)]
        edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            edge_color=[color],
            width=max(lw, 2.0),
            style=ls,
            arrows=True,
            arrowsize=16,
            arrowstyle="-|>",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=path,
            node_size=58,
            node_color=[color] * len(path),
            alpha=0.92,
            linewidths=0.6,
            edgecolors="white",
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[source, target],
        node_size=120,
        node_color="gold",
        edgecolors="black",
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels={source: "Src", target: "Dst"},
        font_size=10,
        font_weight="bold",
    )

    plt.title(
        f"K Shortest Simple Paths (length_km): {source} → {target}  (K={n_paths})",
        fontsize=16,
        fontweight="bold",
    )

    handles = []
    for i, p in enumerate(paths):
        km = path_length_km(G, p)
        ls = _PATH_LINESTYLES[i % len(_PATH_LINESTYLES)]
        lw = 4.2 - i * 0.55
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=path_colors[i],
                linewidth=max(lw, 2.0),
                linestyle=ls,
                label=f"Path {i + 1}: {len(p) - 1} hops, {km:.1f} km",
            )
        )
    plt.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.92)
    plt.axis("off")
    plt.tight_layout()

    safe_src = str(source).replace("/", "_")
    safe_dst = str(target).replace("/", "_")
    filepath = os.path.join(output_dir, f"ksp_{safe_src}_{safe_dst}.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"已保存: {filepath}")
    return filepath


def _normalize_service_id(raw):
    """将 INT081、int-219 等写法规范为 INT-XXX。"""
    s = raw.strip().upper().replace("INT_", "INT-")
    if s.startswith("INT-") and len(s) > 4 and s[4:].isdigit():
        return f"INT-{int(s[4:]):03d}"
    if s.startswith("INT") and len(s) > 3 and s[3:].isdigit():
        return f"INT-{int(s[3:]):03d}"
    return s


def main():
    parser = argparse.ArgumentParser(description="绘制 K 条最短简单路径示意图")
    parser.add_argument("--source", default=None, help="起点 Site Ref.（与 --service 二选一）")
    parser.add_argument("--target", default=None, help="终点 Site Ref.")
    parser.add_argument(
        "--service",
        default=None,
        metavar="INT-XXX",
        help="按流量业务 ID 取起讫点，例如 INT-081、INT-335",
    )
    parser.add_argument("--k", type=int, default=5, help="路径条数上限")
    parser.add_argument(
        "--output-dir", default="output_images", help="输出目录"
    )
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    loader = TefnetLoader(
        os.path.join(root, "data/tefnet_nodes.csv"),
        os.path.join(root, "data/tefnet_links.csv"),
        os.path.join(root, "data/tefnet_traffic.csv"),
    )
    G = loader.load_topology()
    demands = loader.load_traffic_demands()

    source, target = args.source, args.target
    if args.service:
        sid = _normalize_service_id(args.service)
        found = next((d for d in demands if d["id"] == sid), None)
        if found is None:
            print(f"未找到业务: {args.service}（规范化为 {sid}）")
            sys.exit(1)
        source, target = found["source"], found["target"]
        print(f"业务 {found['id']}: {source} → {target}")
    elif source is None or target is None:
        source, target = DEFAULT_SRC, DEFAULT_DST
        print(f"使用默认高对比 OD（INT-081 同对）: {source} → {target}")

    if source not in G or target not in G:
        print(f"节点不在图中: source={source}, target={target}")
        sys.exit(1)

    out = draw_ksp_figure(
        G, source, target, k=args.k, output_dir=os.path.join(root, args.output_dir)
    )
    if out and args.service:
        sid = _normalize_service_id(args.service)
        alt = os.path.join(
            os.path.dirname(out),
            f"ksp_{sid.replace('-', '_')}.png",
        )
        try:
            import shutil

            shutil.copy2(out, alt)
            print(f"已复制为: {alt}")
        except OSError:
            pass


if __name__ == "__main__":
    main()
