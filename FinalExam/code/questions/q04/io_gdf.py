from __future__ import annotations

from pathlib import Path
import pandas as pd

def read_gdf(path: Path, encoding: str = "utf-8"):
    """
    Read a .gdf into (nodes_df, edges_df).
    Robust to minor format inconsistencies.
    """
    mode = None
    node_cols, edge_cols = None, None
    nodes, edges = [], []

    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("nodedef>"):
                mode = "node"
                hdr = line[len("nodedef>"):]
                node_cols = [c.split()[0] for c in hdr.split(",")]
                continue

            if line.startswith("edgedef>"):
                mode = "edge"
                hdr = line[len("edgedef>"):]
                edge_cols = [c.split()[0] for c in hdr.split(",")]
                continue

            if mode == "node":
                parts = line.split(",")
                if node_cols is None or len(parts) != len(node_cols):
                    continue
                nodes.append(parts)

            elif mode == "edge":
                parts = line.split(",")
                if edge_cols is None or len(parts) != len(edge_cols):
                    continue
                edges.append(parts)

    if node_cols is None or edge_cols is None:
        raise ValueError(f"Invalid GDF format: {path}")

    nodes_df = pd.DataFrame(nodes, columns=node_cols)
    edges_df = pd.DataFrame(edges, columns=edge_cols)
    return nodes_df, edges_df

def clean_nodes(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge duplicate node ids (name). 
    - Sum: views, posts, positive, negative, neutral
    - Max: followers, importance
    - Label: most frequent
    """
    df = nodes_df.copy()
    df["name"] = df["name"].astype(str)

    for c in [c for c in df.columns if c not in ["name", "label"]]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def agg_label(s):
        s = s.dropna().astype(str)
        return s.value_counts().idxmax() if len(s) else None

    agg_dict = {}
    for c in ["views", "posts", "positive", "negative", "neutral"]:
        if c in df.columns:
            agg_dict[c] = "sum"
    if "followers" in df.columns:
        agg_dict["followers"] = "max"
    if "importance" in df.columns:
        agg_dict["importance"] = "max"

    out = df.groupby("name", as_index=False).agg({**agg_dict, "label": agg_label})
    return out

def clean_edges(edges_df: pd.DataFrame) -> pd.DataFrame:
    df = edges_df.copy()
    df["node1"] = df["node1"].astype(str)
    df["node2"] = df["node2"].astype(str)

    if "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
    else:
        df["weight"] = 1.0

    return df[["node1", "node2", "weight"]]
