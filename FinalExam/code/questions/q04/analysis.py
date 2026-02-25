from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import networkx as nx

try:
    from networkx.algorithms.community import louvain_communities
except Exception as e:
    raise ImportError(
        "Your NetworkX version does not expose louvain_communities. "
        "Upgrade networkx (>= 3.x recommended)."
    ) from e

from .config import SEED
from .io_gdf import read_gdf, clean_nodes, clean_edges

def build_undirected_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """
    Build an undirected weighted graph, aggregating parallel edges deterministically.
    """
    G = nx.Graph()

    attrs = nodes_df.set_index("name").to_dict(orient="index")
    G.add_nodes_from(attrs.keys())
    nx.set_node_attributes(G, attrs)

    u = edges_df["node1"].to_numpy()
    v = edges_df["node2"].to_numpy()
    a = np.minimum(u, v)
    b = np.maximum(u, v)

    tmp = pd.DataFrame({"u": a, "v": b, "weight": edges_df["weight"].to_numpy()})
    tmp = tmp.sort_values(["u", "v"], kind="mergesort")
    agg = tmp.groupby(["u", "v"], sort=True)["weight"].sum().reset_index()

    G.add_weighted_edges_from(
        agg[["u", "v", "weight"]].itertuples(index=False, name=None),
        weight="weight",
    )
    return G

def load_platform_graph(extract_dir: Path, platform: str, gdf_name_fmt: str):
    """
    Loads and cleans GDF, returns:
      (G, nodes_df, edges_df, gdf_path)
    """
    gdf_path = extract_dir / platform / gdf_name_fmt.format(platform=platform)
    nodes_raw, edges_raw = read_gdf(gdf_path)
    nodes = clean_nodes(nodes_raw)
    edges = clean_edges(edges_raw)
    G = build_undirected_graph(nodes, edges)
    return G, nodes, edges, gdf_path

def stable_sort_communities(comms):
    def key(c):
        return (-len(c), min(map(str, c)))
    return sorted(comms, key=key)

def run_louvain_and_filter(G: nx.Graph, resolution: float = 1.0, seed: int = SEED):
    """
    Runs Louvain and returns:
      comms: list[set[node]]
      summary: DataFrame of communities with size >= 5% of nodes
      min_size: ceil(0.05 * n_nodes)
    """
    comms = louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    comms = stable_sort_communities(comms)

    n = G.number_of_nodes()
    min_size = math.ceil(0.05 * n)

    big = [(i, c) for i, c in enumerate(comms) if len(c) >= min_size]
    big = sorted(big, key=lambda x: (-len(x[1]), x[0]))

    summary = pd.DataFrame(
        {
            "community_id": [i for i, _ in big],
            "size": [len(c) for _, c in big],
            "percent": [len(c) / n for _, c in big],
        }
    )
    return comms, summary, min_size

def choose_resolution_for_instagram(G: nx.Graph, seed: int = SEED):
    """
    Notebook heuristic: try smaller resolutions until a community reaches >= 5% size.
    """
    candidates = [1.0, 0.5, 0.2, 0.1]
    n = G.number_of_nodes()
    min_size = math.ceil(0.05 * n)

    for r in candidates:
        comms = louvain_communities(G, weight="weight", resolution=r, seed=seed)
        if max(len(c) for c in comms) >= min_size:
            return r
    return 1.0

def attach_community_id(G: nx.Graph, comms):
    node2cid = {}
    for cid, c in enumerate(comms):
        for node in c:
            node2cid[node] = cid
    nx.set_node_attributes(G, node2cid, "louvain_cid")
    return node2cid

def export_for_gephi(G: nx.Graph, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, out_path)

def top_accounts_tables(G: nx.Graph, comm_nodes, top_k: int = 10):
    """
    Returns two dataframes: (top weighted degree, top pagerank)
    """
    H = G.subgraph(comm_nodes).copy()

    wdeg = dict(H.degree(weight="weight"))
    top_wdeg = sorted(wdeg.items(), key=lambda x: x[1], reverse=True)[:top_k]

    pr = nx.pagerank(H, weight="weight")
    top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def to_df(pairs, colname):
        df = pd.DataFrame(pairs, columns=["node_id", colname])
        labels = nx.get_node_attributes(H, "label")
        df["username"] = df["node_id"].map(labels)
        followers = nx.get_node_attributes(H, "followers")
        df["followers"] = df["node_id"].map(followers)
        return df[["node_id", "username", "followers", colname]]

    return to_df(top_wdeg, "weighted_degree"), to_df(top_pr, "pagerank")
