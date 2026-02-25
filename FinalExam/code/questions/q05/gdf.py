from __future__ import annotations

import io
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import networkx as nx


def _read_gdf_sections(path: Path):
    """Splits a .gdf file into node and edge sections (raw text)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    m_node = re.search(r"(?im)^nodedef>\s*(.+)$", text)
    m_edge = re.search(r"(?im)^edgedef>\s*(.+)$", text)
    if not m_node or not m_edge:
        raise ValueError(f"Could not find nodedef/edgedef in {path}")

    node_header = m_node.group(1).strip()
    edge_header = m_edge.group(1).strip()

    node_body = text[m_node.end():m_edge.start()].strip("\n")
    edge_body = text[m_edge.end():].strip("\n")

    return node_header, node_body, edge_header, edge_body


def _gdf_header_to_columns(header: str):

    cols = []
    for item in header.split(","):
        col = item.strip().split(" ")[0].strip()
        cols.append(col)
    return cols


def load_gdf_as_digraph(path: Path, weight_col: str = "weight") -> nx.DiGraph:
    """Loads a GDF as a directed weighted DiGraph."""
    node_header, node_body, edge_header, edge_body = _read_gdf_sections(path)

    node_cols = _gdf_header_to_columns(node_header)
    edge_cols = _gdf_header_to_columns(edge_header)

    nodes_df = pd.read_csv(io.StringIO(node_body), header=None, names=node_cols, sep=",", engine="python")
    edges_df = pd.read_csv(io.StringIO(edge_body), header=None, names=edge_cols, sep=",", engine="python")

    src_col = "node1" if "node1" in edges_df.columns else edges_df.columns[0]
    dst_col = "node2" if "node2" in edges_df.columns else edges_df.columns[1]


    if "directed" in edges_df.columns:
        directed_mask = edges_df["directed"].astype(str).str.lower().isin(["true", "1", "t", "yes"])
        if directed_mask.any():
            edges_df = edges_df[directed_mask].copy()

    if weight_col in edges_df.columns:
        w = edges_df[weight_col].apply(lambda x: float(x) if pd.notna(x) else 1.0)
    else:
        w = pd.Series([1.0] * len(edges_df), index=edges_df.index)

    G = nx.DiGraph()

    node_id_col = "name" if "name" in nodes_df.columns else nodes_df.columns[0]
    for _, row in nodes_df.iterrows():
        nid = row[node_id_col]
        G.add_node(nid, **row.to_dict())


    agg = defaultdict(float)
    for u, v, ww in zip(edges_df[src_col].values, edges_df[dst_col].values, w.values):
        if pd.isna(u) or pd.isna(v) or u == v:
            continue
        agg[(u, v)] += float(ww)

    for (u, v), ww in agg.items():
        G.add_edge(u, v, weight=ww)

    return G


def discover_platform_gdfs(data_dir: Path) -> dict:
    """Returns dict platform -> chosen .gdf path (largest file per platform)."""
    out = {}
    for platform in ["twitter", "telegram", "instagram"]:
        cand = sorted((data_dir / platform).glob("*.gdf"))
        if cand:
            cand.sort(key=lambda p: p.stat().st_size, reverse=True)
            out[platform] = cand[0]
    return out
