from __future__ import annotations

import time
from pathlib import Path
from collections import defaultdict

import pandas as pd
import networkx as nx


def project_to_undirected_sum_weights(G_dir: nx.DiGraph) -> nx.Graph:
    """Undirected projection: w(u,v) = w(u->v) + w(v->u)."""
    Gu = nx.Graph()
    Gu.add_nodes_from(G_dir.nodes(data=True))
    w = defaultdict(float)
    for u, v, data in G_dir.edges(data=True):
        if u == v:
            continue
        key = (u, v) if str(u) <= str(v) else (v, u)
        w[key] += float(data.get("weight", 1.0))
    for (u, v), ww in w.items():
        Gu.add_edge(u, v, weight=ww)
    return Gu


def _to_node2comm_from_sets(communities) -> dict:
    node2c = {}
    for cid, cset in enumerate(communities):
        for n in cset:
            node2c[n] = cid
    return node2c


def run_louvain(G: nx.Graph, seed: int = 42) -> dict:
    comms = nx.community.louvain_communities(G, weight="weight", seed=int(seed))
    return _to_node2comm_from_sets(comms)


def run_lpa_undirected(G_und: nx.Graph, seed: int = 42) -> dict:
    comms = nx.community.asyn_lpa_communities(G_und, weight="weight", seed=int(seed))
    return _to_node2comm_from_sets(comms)


def run_infomap_fast(G_dir: nx.DiGraph, seed: int = 42, preset: str = "fast"):
    """Robust Infomap runner with speed presets and safe node-id handling."""
    try:
        from infomap import Infomap
    except Exception as e:
        raise ImportError("Infomap is not installed. Install via: pip install infomap") from e

    if preset == "ultrafast":
        kwargs = dict(directed=True, two_level=True, silent=True, seed=int(seed),
                      num_trials=1, core_loop_limit=5, fast_hierarchical_solution=3,
                      inner_parallelization=True, no_self_links=True)
        option_str = f"--directed --two-level --silent --seed {int(seed)} --num-trials 1 --core-loop-limit 5 --fast-hierarchical-solution 3 --no-self-links"
    elif preset == "fast":
        kwargs = dict(directed=True, two_level=True, silent=True, seed=int(seed),
                      num_trials=3, core_loop_limit=10, fast_hierarchical_solution=3,
                      inner_parallelization=True, no_self_links=True)
        option_str = f"--directed --two-level --silent --seed {int(seed)} --num-trials 3 --core-loop-limit 10 --fast-hierarchical-solution 3 --no-self-links"
    else:
        kwargs = dict(directed=True, two_level=False, silent=True, seed=int(seed),
                      num_trials=10, no_self_links=True)
        option_str = f"--directed --silent --seed {int(seed)} --num-trials 10 --no-self-links"

    try:
        im = Infomap(**kwargs)
    except TypeError:
        im = Infomap(option_str)


    id2node = None
    added = False
    if hasattr(im, "add_networkx_graph"):
        try:
            im.add_networkx_graph(G_dir, weight="weight")
            added = True
        except Exception:
            added = False

    if not added:
        nodes = list(G_dir.nodes())
        node2id = {n: i for i, n in enumerate(nodes)}
        for u, v, data in G_dir.edges(data=True):
            if u == v:
                continue
            im.addLink(int(node2id[u]), int(node2id[v]), float(data.get("weight", 1.0)))
        id2node = {i: n for n, i in node2id.items()}

    t0 = time.perf_counter()
    im.run()
    t1 = time.perf_counter()

    modules = im.get_modules()
    if id2node is not None:
        modules = {id2node[int(k)]: int(v) for k, v in modules.items()}

    meta = dict(
        seconds=float(t1 - t0),
        num_modules=int(getattr(im, "num_top_modules", len(set(modules.values())))),
        codelength=float(getattr(im, "codelength", float("nan"))),
        preset=preset,
    )
    return modules, meta


def _topk_communities(node2comm: dict, k: int = 10):
    buckets = defaultdict(list)
    for n, cid in node2comm.items():
        buckets[cid].append(n)
    comms = [(cid, len(nodes), set(nodes)) for cid, nodes in buckets.items()]
    comms.sort(key=lambda x: x[1], reverse=True)
    return comms[:k]


def _best_jaccard_matches(topA, topB):
    rows = []
    for cidA, sizeA, setA in topA:
        best = None
        for cidB, sizeB, setB in topB:
            inter = len(setA & setB)
            union = len(setA | setB)
            j = (inter / union) if union else 0.0
            if best is None or j > best["jaccard"]:
                best = dict(
                    cid_A=cidA, size_A=sizeA,
                    cid_B=cidB, size_B=sizeB,
                    intersection=inter,
                    jaccard=j,
                    cover_A=(inter / sizeA) if sizeA else 0.0,
                    cover_B=(inter / sizeB) if sizeB else 0.0,
                )
        rows.append(best)
    return rows


def save_node2comm(node2comm: dict, path: Path, colname: str = "community_id") -> None:
    df = pd.DataFrame({"node": list(node2comm.keys()), colname: list(node2comm.values())})
    df.to_csv(path, index=False, compression="gzip")


def maybe_export_gexf(G: nx.Graph, node_attrs: dict, path: Path) -> None:

    Gx = G.copy()
    for attr_name, mapping in node_attrs.items():
        nx.set_node_attributes(Gx, mapping, attr_name)
    nx.write_gexf(Gx, path, prettyprint=False)
