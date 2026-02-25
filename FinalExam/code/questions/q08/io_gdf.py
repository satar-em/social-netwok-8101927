from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd

def read_gdf_from_zip(zip_path: str | Path, inner_path: str):
    """Read a .gdf file from inside a zip and return (nodes_df, edges_df)."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        raw = z.read(inner_path).decode("utf-8", errors="replace")

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() != ""]
    nodestart = None
    edgestart = None
    for i, ln in enumerate(lines):
        if ln.lower().startswith("nodedef"):
            nodestart = i
        if ln.lower().startswith("edgedef"):
            edgestart = i
            break

    if nodestart is None or edgestart is None:
        raise ValueError("Could not find nodedef/edgedef in GDF file")

    node_header = lines[nodestart].split(">")[1]
    edge_header = lines[edgestart].split(">")[1]
    node_cols = [c.split()[0] for c in node_header.split(",")]
    edge_cols = [c.split()[0] for c in edge_header.split(",")]

    def parse_line(line: str, ncols: int):

        parts = [p.strip() for p in re.split(r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", line)]
        parts = [p[1:-1] if len(p) >= 2 and p[0] == '"' and p[-1] == '"' else p for p in parts]

        if len(parts) < ncols:
            parts += [""] * (ncols - len(parts))
        elif len(parts) > ncols:
            parts = parts[: ncols - 1] + [",".join(parts[ncols - 1 :])]

        return parts

    node_lines = lines[nodestart + 1 : edgestart]
    edge_lines = lines[edgestart + 1 :]

    ndf = pd.DataFrame([parse_line(ln, len(node_cols)) for ln in node_lines], columns=node_cols)
    edf = pd.DataFrame([parse_line(ln, len(edge_cols)) for ln in edge_lines], columns=edge_cols)
    return ndf, edf
