"""
Core analysis utilities for HW Q2: graph building and metrics.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import networkx as nx

from .io_gdf import read_gdf


def to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    edges_df = edges_df.copy()
    edges_df["node1"] = edges_df["node1"].astype(str)
    edges_df["node2"] = edges_df["node2"].astype(str)
    edges_df["weight"] = pd.to_numeric(edges_df["weight"], errors="coerce").fillna(1).astype(float)

    G = nx.DiGraph()


    for _, r in nodes_df.iterrows():
        nid = str(r["name"])
        G.add_node(nid, **r.to_dict())


    for _, r in edges_df.iterrows():
        u, v, w = str(r["node1"]), str(r["node2"]), float(r["weight"])
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    return G


def topk_from_dict(d: dict, k=10, node_to_label=None) -> pd.DataFrame:
    items = sorted(d.items(), key=lambda x: (-x[1], str(x[0])))[:k]
    rows = []
    for node, score in items:
        rows.append({
            "node_id": str(node),
            "username": node_to_label.get(str(node)) if node_to_label else None,
            "score": float(score),
        })
    return pd.DataFrame(rows)


def compute_metrics(G: nx.DiGraph, id_to_label: dict, k=10):
    indeg = dict(G.in_degree())
    indeg_w = dict(G.in_degree(weight="weight"))

    pr = nx.pagerank(G, weight="weight")


    try:
        hubs, auth = nx.hits(G, max_iter=1000, tol=1e-8, normalized=True)
    except nx.PowerIterationFailedConvergence:
        hubs, auth = nx.hits(G, max_iter=5000, tol=1e-8, normalized=True)

    return {
        "in_degree": topk_from_dict(indeg, k, id_to_label),
        "in_degree_weighted": topk_from_dict(indeg_w, k, id_to_label),
        "pagerank": topk_from_dict(pr, k, id_to_label),
        "hubs": topk_from_dict(hubs, k, id_to_label),
        "authorities": topk_from_dict(auth, k, id_to_label),
    }


def run_platform(extract_dir: str, platform: str, gdf_suffix: str, k: int = 10):
    """
    Load platform gdf, build weighted directed graph, compute top-k metrics.
    """
    gdf_path = os.path.join(extract_dir, platform, f"{platform}-{gdf_suffix}.gdf")
    nodes_df, edges_df = read_gdf(gdf_path)


    nodes_df = to_numeric(nodes_df, ["views","followers","importance","posts","positive","negative","neutral"])

    id_to_label = dict(zip(nodes_df["name"].astype(str), nodes_df["label"].astype(str)))
    G = build_graph(nodes_df, edges_df)

    metrics = compute_metrics(G, id_to_label, k=k)
    return G, id_to_label, metrics


def overlap_analysis(metrics_dict):
    """
    Compute overlaps of Top-k lists among Hub/Authority/PageRank/InDegree.
    Returns: (sets_dict, intersection_all4, pairwise_df)
    """
    sets = {
        "Hub": set(metrics_dict["hubs"]["node_id"]),
        "Authority": set(metrics_dict["authorities"]["node_id"]),
        "PageRank": set(metrics_dict["pagerank"]["node_id"]),
        "InDegree": set(metrics_dict["in_degree"]["node_id"]),
    }

    keys = list(sets.keys())
    rows = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a, b = keys[i], keys[j]
            inter = sets[a] & sets[b]
            rows.append({
                "A": a, "B": b,
                "intersection_size": len(inter),
                "intersection_nodes": sorted(list(inter))[:20],
            })

    all4 = set.intersection(*sets.values())
    pair = pd.DataFrame(rows).sort_values("intersection_size", ascending=False)
    return sets, all4, pair


def save_outputs(platform_results: dict, output_dir: str) -> None:
    """
    Save the same CSV outputs as the notebook.
    """
    os.makedirs(output_dir, exist_ok=True)

    for plat, payload in platform_results.items():
        m = payload["metrics"]
        m["in_degree"].to_csv(os.path.join(output_dir, f"{plat}_top10_indegree.csv"), index=False)
        m["pagerank"].to_csv(os.path.join(output_dir, f"{plat}_top10_pagerank.csv"), index=False)
        m["authorities"].to_csv(os.path.join(output_dir, f"{plat}_top10_authority.csv"), index=False)
        m["hubs"].to_csv(os.path.join(output_dir, f"{plat}_top10_hub.csv"), index=False)

        _, _, pair = overlap_analysis(m)
        pair.to_csv(os.path.join(output_dir, f"{plat}_overlaps.csv"), index=False)
