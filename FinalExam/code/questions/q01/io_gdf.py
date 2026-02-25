from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

import pandas as pd


def read_gdf(path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read a GDF graph file into (nodes, edges) dataframes.

    The notebook expects files formatted with 'nodedef>' and 'edgedef>' headers.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    nodedef_idx = None
    edgedef_idx = None
    for i, line in enumerate(lines):
        if line.startswith("nodedef>"):
            nodedef_idx = i
        if line.startswith("edgedef>"):
            edgedef_idx = i
            break
    if nodedef_idx is None or edgedef_idx is None:
        raise ValueError(f"Invalid GDF structure: {path}")

    nodedef = lines[nodedef_idx][len("nodedef>"):]
    node_cols = [c.split()[0].strip() for c in nodedef.split(",")]
    node_lines = lines[nodedef_idx + 1 : edgedef_idx]
    node_rows = [row for row in csv.reader(node_lines) if row]
    nodes = pd.DataFrame(node_rows, columns=node_cols)


    for col in ["name", "views", "followers", "posts", "positive", "negative", "neutral"]:
        if col in nodes.columns:
            nodes[col] = pd.to_numeric(nodes[col], errors="coerce")
    if "importance" in nodes.columns:
        nodes["importance"] = pd.to_numeric(nodes["importance"], errors="coerce")

    edgedef = lines[edgedef_idx][len("edgedef>"):]
    edge_cols = [c.split()[0].strip() for c in edgedef.split(",")]
    edge_lines = lines[edgedef_idx + 1 :]
    edge_rows = [row for row in csv.reader(edge_lines) if row]
    edges = pd.DataFrame(edge_rows, columns=edge_cols)

    for col in ["node1", "node2", "weight"]:
        if col in edges.columns:
            edges[col] = pd.to_numeric(edges[col], errors="coerce")

    return nodes, edges
