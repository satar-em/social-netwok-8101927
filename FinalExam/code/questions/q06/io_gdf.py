"""GDF parsing + DiGraph building (ported from Q6 notebook)."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import networkx as nx


def parse_gdf(path: Path):
    """Parse a GDF file into (nodes_df, edges_df)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    edge_header_idx = None
    for i, line in enumerate(lines):
        if line.lower().startswith("edgedef>"):
            edge_header_idx = i
            break
    if edge_header_idx is None:
        raise ValueError("No edgedef> section found in GDF.")

    node_header = lines[0]
    node_lines = lines[1:edge_header_idx]
    edge_header = lines[edge_header_idx]
    edge_lines = lines[edge_header_idx + 1:]

    node_cols = node_header.split(">")[1].split(",")
    nodes = [l.split(",") for l in node_lines if l.strip()]
    nodes_df = pd.DataFrame(nodes, columns=node_cols)


    rename_map = {}
    for c in nodes_df.columns:
        if c.startswith("label"):
            rename_map[c] = "label"
    nodes_df = nodes_df.rename(columns=rename_map)


    for c in nodes_df.columns:
        if c == "name":
            nodes_df[c] = nodes_df[c].astype(str)

    int_cols = ["views", "followers", "posts", "positive", "negative", "neutral"]
    float_cols = ["importance"]
    for c in int_cols:
        if c in nodes_df.columns:
            nodes_df[c] = pd.to_numeric(nodes_df[c], errors="coerce").fillna(0).astype("int64")
    for c in float_cols:
        if c in nodes_df.columns:
            nodes_df[c] = pd.to_numeric(nodes_df[c], errors="coerce")


    edge_cols = edge_header.split(">")[1].split(",")
    edges = [l.split(",") for l in edge_lines if l.strip()]
    edges_df = pd.DataFrame(edges, columns=edge_cols)

    edges_df = edges_df.rename(columns={"node1": "src", "node2": "dst", "weight DOUBLE": "weight"})
    if "weight" not in edges_df.columns and "weight DOUBLE" in edges_df.columns:
        edges_df = edges_df.rename(columns={"weight DOUBLE": "weight"})
    if "directed BOOLEAN" in edges_df.columns:
        edges_df = edges_df.rename(columns={"directed BOOLEAN": "directed"})

    edges_df["src"] = edges_df["src"].astype(str)
    edges_df["dst"] = edges_df["dst"].astype(str)
    edges_df["weight"] = pd.to_numeric(edges_df["weight"], errors="coerce").fillna(1.0)
    if "directed" in edges_df.columns:
        edges_df["directed"] = edges_df["directed"].astype(str).str.lower().isin(["true", "1", "yes", "y"])

    return nodes_df, edges_df


def build_digraph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    """Build a weighted directed graph (aggregates repeated edges)."""
    G = nx.DiGraph()

    node_attr_cols = [c for c in nodes_df.columns if c != "name"]
    for _, r in nodes_df.iterrows():
        nid = r["name"]
        attrs = {c: r[c] for c in node_attr_cols}
        G.add_node(nid, **attrs)

    agg = edges_df.groupby(["src", "dst"], as_index=False)["weight"].sum()
    G.add_weighted_edges_from(
        agg[["src", "dst", "weight"]].itertuples(index=False, name=None),
        weight="weight",
    )
    return G
