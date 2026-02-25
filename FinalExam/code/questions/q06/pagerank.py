"""PageRank utilities (ported from Q6 notebook)."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import networkx as nx

from .io_gdf import parse_gdf, build_digraph


def compute_pagerank(G: nx.DiGraph, alpha: float = 0.85) -> dict:
    """Compute weighted PageRank."""

    return nx.pagerank(G, alpha=alpha, weight="weight", tol=1e-8, max_iter=200)


def topk_pagerank_for_platform(platform_dir: Path, k: int = 1000, alpha: float = 0.85):
    """Load the platform GDF, build graph, compute top-k PageRank.

    Returns (pr_df, G) where pr_df includes node attributes from GDF.
    """
    gdf_path = next(iter(platform_dir.glob("*.gdf")))
    nodes_df, edges_df = parse_gdf(gdf_path)
    G = build_digraph(nodes_df, edges_df)

    pr = compute_pagerank(G, alpha=alpha)
    pr_df = pd.DataFrame({"node_id": list(pr.keys()), "pagerank": list(pr.values())})
    pr_df = pr_df.sort_values("pagerank", ascending=False).head(k).reset_index(drop=True)

    nodes_meta = nodes_df.rename(columns={"name": "node_id"})
    pr_df = pr_df.merge(nodes_meta, on="node_id", how="left")
    return pr_df, G
