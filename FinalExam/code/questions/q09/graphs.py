from __future__ import annotations

from collections import defaultdict
import networkx as nx
import pandas as pd

from .config import TOP_N
from .gdf import read_gdf


def load_gdfs(gdf_paths: dict[str, 'Path']):
    nodes_dfs, edges_dfs = {}, {}
    for plat, p in gdf_paths.items():
        ndf, edf = read_gdf(p)
        nodes_dfs[plat] = ndf
        edges_dfs[plat] = edf
    return nodes_dfs, edges_dfs


def build_working_graph(platform: str, nodes_dfs: dict, edges_dfs: dict):
    ndf = nodes_dfs[platform].copy()
    edf = edges_dfs[platform].copy()


    ndf["name"] = ndf["name"].astype(str).str.strip()
    ndf["name"] = ndf["name"].str.replace(r"\.0$", "", regex=True)

    if "label" not in ndf.columns:
        alt = next((c for c in ndf.columns if c.startswith("label")), None)
        if alt is not None:
            ndf = ndf.rename(columns={alt: "label"})
        elif "username" in ndf.columns:
            ndf = ndf.rename(columns={"username": "label"})
        else:
            ndf["label"] = ndf["name"]

    ndf["label"] = ndf["label"].astype(str).str.strip().str.lower().str.lstrip("@")


    if "importance" in ndf.columns:
        ndf = ndf.sort_values("importance", ascending=False, na_position="last")
    elif "followers" in ndf.columns:
        ndf = ndf.sort_values("followers", ascending=False, na_position="last")

    ndf = ndf.dropna(subset=["name"]).copy()
    ndf = ndf.drop_duplicates(subset=["name"], keep="first").copy()

    top_nodes = set(ndf.head(TOP_N[platform])["name"].tolist())

    edf["node1"] = edf["node1"].astype(str)
    edf["node2"] = edf["node2"].astype(str)
    edf = edf[edf["node1"].isin(top_nodes) & edf["node2"].isin(top_nodes)].copy()


    Gd = nx.DiGraph()

    keep_cols = [c for c in ["label", "followers", "posts", "importance"] if c in ndf.columns]
    attr_map = (
        ndf.loc[ndf["name"].isin(top_nodes), ["name"] + keep_cols]
           .drop_duplicates(subset=["name"], keep="first")
           .set_index("name")[keep_cols]
           .to_dict(orient="index")
    )
    Gd.add_nodes_from([(nid, attr_map.get(nid, {})) for nid in top_nodes])

    if "weight" not in edf.columns:
        edf["weight"] = 1.0
    Gd.add_weighted_edges_from([(r["node1"], r["node2"], float(r.get("weight", 1.0))) for _, r in edf.iterrows()])


    Gu = nx.Graph()
    Gu.add_nodes_from(Gd.nodes(data=True))
    w = defaultdict(float)
    for u, v, data in Gd.edges(data=True):
        if u == v:
            continue
        a, b = (u, v) if u <= v else (v, u)
        w[(a, b)] += float(data.get("weight", 1.0))
    for (a, b), ww in w.items():
        Gu.add_edge(a, b, weight=ww)

    return Gd, Gu


def attach_seed_to_graph(Gd: nx.DiGraph, seed_df: pd.DataFrame) -> dict:
    seed_df = seed_df.copy()
    seed_df["user_key"] = seed_df["user_key"].astype(str).str.strip().str.lower().str.lstrip("@")
    seed_map = seed_df.set_index("user_key")["seed_label"].to_dict()
    score_map = seed_df.set_index("user_key")["seed_score"].to_dict()

    seed_on_node, seed_score_on_node = {}, {}
    for nid, data in Gd.nodes(data=True):
        lbl = str(data.get("label", "")).strip().lower().lstrip("@")
        seed_on_node[nid] = int(seed_map.get(lbl, 0))
        seed_score_on_node[nid] = float(score_map.get(lbl, 0.0))

    nx.set_node_attributes(Gd, seed_on_node, "seed_label")
    nx.set_node_attributes(Gd, seed_score_on_node, "seed_score")
    return seed_on_node
