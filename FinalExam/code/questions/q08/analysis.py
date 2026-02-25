from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities

from .io_gdf import read_gdf_from_zip
from .text_utils import normalize_handle


def build_digraph(ndf: pd.DataFrame, edf: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()

    for _, row in ndf.iterrows():
        nid = str(row["name"])
        label = str(row.get("label", "")).strip()
        G.add_node(nid, label=label)

    for _, row in edf.iterrows():
        u = str(row["node1"])
        v = str(row["node2"])
        try:
            w = float(row.get("weight", 1) or 1)
        except Exception:
            w = 1.0

        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    return G


def compute_louvain(G: nx.DiGraph, seed: int = 42):
    UG = G.to_undirected()
    comms = louvain_communities(UG, weight="weight", seed=seed, resolution=1)
    node2comm = {}
    for cid, nodes in enumerate(comms):
        for n in nodes:
            node2comm[n] = cid
    sizes = sorted([len(c) for c in comms], reverse=True)
    return node2comm, sizes


def compute_pagerank(G: nx.DiGraph):
    return nx.pagerank(G, alpha=0.85, weight="weight", max_iter=200, tol=1e-6)


@dataclass
class GraphIndex:
    graphs: dict[str, nx.DiGraph]
    node2comm: dict[str, dict[str, int]]
    pagerank: dict[str, dict[str, float]]
    label2comm: dict[str, dict[str, int]]
    label2pr: dict[str, dict[str, float]]


def build_graph_index(data_zip: str | Path, gdf_paths: dict[str, str], seed: int = 42) -> GraphIndex:
    data_zip = Path(data_zip)

    graphs: dict[str, nx.DiGraph] = {}
    node2comm: dict[str, dict[str, int]] = {}
    pagerank: dict[str, dict[str, float]] = {}
    label2comm: dict[str, dict[str, int]] = {}
    label2pr: dict[str, dict[str, float]] = {}

    for plat, inner in gdf_paths.items():
        ndf, edf = read_gdf_from_zip(data_zip, inner)
        G = build_digraph(ndf, edf)
        graphs[plat] = G

        node2comm[plat], _ = compute_louvain(G, seed=seed)
        pagerank[plat] = compute_pagerank(G)


        rows = []
        for n, data in G.nodes(data=True):
            label = (data.get("label") or "").strip()
            rows.append(
                (
                    n,
                    normalize_handle(label),
                    label,
                    pagerank[plat].get(n, 0.0),
                    node2comm[plat].get(n, -1),
                )
            )
        df_nodes = pd.DataFrame(rows, columns=["node_id", "label_lc", "label", "pagerank", "community"])
        best = df_nodes.sort_values("pagerank", ascending=False).drop_duplicates("label_lc")
        label2comm[plat] = dict(zip(best["label_lc"], best["community"]))
        label2pr[plat] = dict(zip(best["label_lc"], best["pagerank"]))

    return GraphIndex(graphs=graphs, node2comm=node2comm, pagerank=pagerank, label2comm=label2comm, label2pr=label2pr)


def attach_graph_features(df: pd.DataFrame, idx: GraphIndex) -> pd.DataFrame:
    df = df.copy()
    df["username_lc"] = df["username"].astype(str).map(normalize_handle)

    comm = []
    pr = []
    for _, r in df.iterrows():
        plat = r["platform"]
        u = r["username_lc"]
        comm.append(idx.label2comm.get(plat, {}).get(u, np.nan))
        pr.append(idx.label2pr.get(plat, {}).get(u, np.nan))

    df["community"] = comm
    df["pagerank"] = pr
    return df


def match_claims(posts_incident: pd.DataFrame, claims: dict, idx: GraphIndex) -> pd.DataFrame:
    claim_frames = []
    for key, info in claims.items():
        pat = info["pattern"]
        m = posts_incident["text_norm"].str.contains(pat, na=False, regex=True)
        df = posts_incident[m].copy()
        df["claim_key"] = key
        df["claim_title"] = info["title"]
        df = attach_graph_features(df, idx)
        claim_frames.append(df)

    if not claim_frames:
        return pd.DataFrame()

    return pd.concat(claim_frames, ignore_index=True)


def claim_examples(claims_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "claim_key",
        "claim_title",
        "platform",
        "dey_day",
        "username",
        "engagement_filled",
        "community",
        "pagerank",
        "link",
        "text_norm",
    ]
    existing = [c for c in cols if c in claims_df.columns]
    return (
        claims_df.sort_values("engagement_filled", ascending=False)
        .groupby("claim_key", as_index=False)
        .head(10)[existing]
    )


def claim_accounts(claims_df: pd.DataFrame) -> pd.DataFrame:
    return (
        claims_df.groupby(["claim_key", "platform", "username_lc"], dropna=False)
        .agg(
            username=("username", "first"),
            n_posts=("text_norm", "size"),
            engagement=("engagement_filled", "sum"),
            community=("community", "first"),
            pagerank=("pagerank", "max"),
        )
        .reset_index()
        .sort_values(["claim_key", "platform", "n_posts", "engagement"], ascending=[True, True, False, False])
    )


def claim_communities(claims_df: pd.DataFrame) -> pd.DataFrame:
    return (
        claims_df.dropna(subset=["community"])
        .groupby(["claim_key", "platform", "community"])
        .agg(n_posts=("text_norm", "size"), n_users=("username_lc", "nunique"), engagement=("engagement_filled", "sum"))
        .reset_index()
        .sort_values(["claim_key", "platform", "n_posts", "engagement"], ascending=[True, True, False, False])
    )


def telegram_multi_claim_communities(comm: pd.DataFrame, topn: int = 25) -> pd.DataFrame:
    tg = comm[comm["platform"] == "telegram"].copy()
    if tg.empty:
        return pd.DataFrame()

    pivot = tg.pivot_table(index="community", columns="claim_key", values="n_posts", fill_value=0, aggfunc="sum")
    pivot["total_posts"] = pivot.sum(axis=1)
    pivot["n_claims"] = (pivot.drop(columns=["total_posts"]) > 0).sum(axis=1)

    multi = pivot.sort_values(["n_claims", "total_posts"], ascending=[False, False]).head(topn)
    return multi


def top_nodes_in_community(idx: GraphIndex, platform: str, community_id: int, k: int = 15) -> pd.DataFrame:
    G = idx.graphs[platform]
    pr = idx.pagerank[platform]
    node2c = idx.node2comm[platform]
    rows = []
    for n, data in G.nodes(data=True):
        if node2c.get(n, -1) == community_id:
            rows.append((data.get("label", ""), pr.get(n, 0.0), n))
    out = pd.DataFrame(rows, columns=["label", "pagerank", "node_id"]).sort_values("pagerank", ascending=False).head(k)
    return out
