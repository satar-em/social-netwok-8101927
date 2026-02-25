"""
GDF parser (nodes + edges).
"""

from __future__ import annotations

import re
import io
import csv
import pandas as pd


def read_gdf(path: str):
    """Read a GDF file into (nodes_df, edges_df)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip("\n") for line in f]

    node_start = None
    edge_start = None
    for i, line in enumerate(lines):
        if line.lower().startswith("nodedef>"):
            node_start = i
        elif line.lower().startswith("edgedef>"):
            edge_start = i
            break

    if node_start is None or edge_start is None:
        raise ValueError("Could not find nodedef> and edgedef> sections")

    node_header = lines[node_start][len("nodedef>"):].strip()
    edge_header = lines[edge_start][len("edgedef>"):].strip()

    def parse_header(h: str):
        cols = []
        for part in h.split(","):
            part = part.strip()
            if not part:
                continue

            m = re.match(r"^([^ ]+)\s+(.+)$", part)
            cols.append(m.group(1) if m else part)
        return cols

    node_cols = parse_header(node_header)
    edge_cols = parse_header(edge_header)

    node_lines = lines[node_start + 1: edge_start]
    edge_lines = lines[edge_start + 1:]

    def parse_csv_lines(raw_lines, cols):
        reader = csv.reader(io.StringIO("\n".join(raw_lines)))
        rows = []
        for r in reader:
            if len(r) == 0:
                continue
            if len(r) < len(cols):
                r = r + [None] * (len(cols) - len(r))
            elif len(r) > len(cols):
                r = r[:len(cols)]
            rows.append(r)
        return pd.DataFrame(rows, columns=cols)

    nodes_df = parse_csv_lines(node_lines, node_cols)
    edges_df = parse_csv_lines(edge_lines, edge_cols)
    return nodes_df, edges_df
