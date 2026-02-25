from __future__ import annotations

from pathlib import Path
import io
import re
import pandas as pd

_GDF_TYPES = {"VARCHAR", "INT", "INTEGER", "LONG", "FLOAT", "DOUBLE", "BOOLEAN"}


def _normalize_gdf_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for col in df.columns:
        c = str(col).strip()
        m = re.match(r"^(.*)\s+(" + "|".join(_GDF_TYPES) + r")\s*$", c, flags=re.IGNORECASE)
        name = m.group(1).strip() if m else c
        name = name.strip('"').strip("'")
        name = name.lower().replace(" ", "_")
        new_cols.append(name)
    out = df.copy()
    out.columns = new_cols
    return out


def read_gdf(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a Gephi/Guess GDF file into (nodes_df, edges_df) with normalized column names."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    edge_idx = None
    for i, line in enumerate(lines):
        if line.startswith("edgedef>"):
            edge_idx = i
            break
    if edge_idx is None:
        raise ValueError("No edgedef> line found in GDF.")

    node_header = lines[0].replace("nodedef>", "")
    edge_header = lines[edge_idx].replace("edgedef>", "")

    node_block = "\n".join([node_header] + lines[1:edge_idx])
    edge_block = "\n".join([edge_header] + lines[edge_idx + 1:])

    nodes = pd.read_csv(io.StringIO(node_block))
    edges = pd.read_csv(io.StringIO(edge_block))

    nodes = _normalize_gdf_columns(nodes)
    edges = _normalize_gdf_columns(edges)

    if "directed" in edges.columns:
        edges["directed"] = (
            edges["directed"].astype(str).str.strip().str.lower().map({"true": True, "false": False}).fillna(True)
        )
    if "weight" in edges.columns:
        edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(1.0)

    if "name" in nodes.columns:
        nodes["name"] = nodes["name"].astype(str)
    if "node1" in edges.columns:
        edges["node1"] = edges["node1"].astype(str)
    if "node2" in edges.columns:
        edges["node2"] = edges["node2"].astype(str)

    return nodes, edges
