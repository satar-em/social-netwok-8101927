from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd

def _split_csv_line(line: str) -> List[str]:
    'Split a GDF CSV line that may contain quoted commas.'
    parts: List[str] = []
    cur = ""
    inq = False
    for ch in line:
        if ch == '"':
            inq = not inq
        elif ch == "," and not inq:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return parts


def read_gdf(gdf_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    'Parse a GDF file into (nodes_df, edges_df). Works with many Gephi-exported GDF variants.'
    text = gdf_path.read_text(encoding="utf-8", errors="ignore")

    nodedef_match = re.search(r"nodedef>.*?\n", text)
    edgedef_match = re.search(r"edgedef>.*?\n", text)
    if not nodedef_match or not edgedef_match:
        raise ValueError(f"Invalid GDF (missing nodedef/edgedef): {gdf_path}")

    nodedef_line = nodedef_match.group(0).strip()
    edgedef_line = edgedef_match.group(0).strip()

    node_block = text[nodedef_match.end() : edgedef_match.start()].strip()
    edge_block = text[edgedef_match.end() :].strip()

    node_cols = [c.split()[0] for c in _split_csv_line(nodedef_line.split(">", 1)[1]) if c]
    edge_cols = [c.split()[0] for c in _split_csv_line(edgedef_line.split(">", 1)[1]) if c]

    node_rows = []
    for line in node_block.splitlines():
        if not line.strip():
            continue
        vals = _split_csv_line(line)
        vals += [""] * (len(node_cols) - len(vals))
        node_rows.append(vals[: len(node_cols)])
    nodes_df = pd.DataFrame(node_rows, columns=node_cols)

    edge_rows = []
    for line in edge_block.splitlines():
        if not line.strip():
            continue
        vals = _split_csv_line(line)
        vals += [""] * (len(edge_cols) - len(vals))
        edge_rows.append(vals[: len(edge_cols)])
    edges_df = pd.DataFrame(edge_rows, columns=edge_cols)


    if "name" in nodes_df.columns:
        nodes_df["name"] = pd.to_numeric(nodes_df["name"], errors="coerce").astype("Int64")

    for col in ["1Node", "2Node", "node1", "node2"]:
        if col in edges_df.columns:
            edges_df[col] = pd.to_numeric(edges_df[col], errors="coerce").astype("Int64")
    for col in ["Weight", "weight"]:
        if col in edges_df.columns:
            edges_df[col] = pd.to_numeric(edges_df[col], errors="coerce")

    return nodes_df, edges_df

