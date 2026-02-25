from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from .config import Q03Config
from .extract import unzip_if_needed, locate_platform_files
from .io_gdf import read_gdf
from .posts import load_platform_excels
from .topics import tfidf_keywords_by_user




def build_undirected_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()


    for _, row in nodes_df.iterrows():
        if pd.isna(row.get("name", np.nan)):
            continue
        nid = int(row["name"])
        attrs = row.to_dict()

        for k, v in list(attrs.items()):
            if isinstance(v, (np.integer,)):
                attrs[k] = int(v)
        G.add_node(nid, **attrs)


    ucol = "1Node" if "1Node" in edges_df.columns else ("node1" if "node1" in edges_df.columns else None)
    vcol = "2Node" if "2Node" in edges_df.columns else ("node2" if "node2" in edges_df.columns else None)
    wcol = "Weight" if "Weight" in edges_df.columns else ("weight" if "weight" in edges_df.columns else None)

    if ucol is None or vcol is None:
        raise ValueError("Could not find edge endpoint columns in GDF edges.")


    for _, row in edges_df.iterrows():
        u = row[ucol]
        v = row[vcol]
        if pd.isna(u) or pd.isna(v):
            continue
        u = int(u); v = int(v)
        w = float(row[wcol]) if (wcol is not None and pd.notna(row[wcol])) else 1.0

        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    return G


def run_kcore_for_platform(base_dir: Path, platform: str) -> Dict[str, object]:
    gdf_path, xlsx_files = locate_platform_files(base_dir, platform)
    nodes_df, edges_df = read_gdf(gdf_path)
    G = build_undirected_graph(nodes_df, edges_df)

    core = nx.core_number(G)
    kmax = max(core.values()) if core else 0


    label_map = {int(r["name"]): r.get("label", None) for _, r in nodes_df.iterrows() if pd.notna(r.get("name", np.nan))}
    out = pd.DataFrame({
        "node": list(core.keys()),
        "k_shell": list(core.values()),
    })
    out["username"] = out["node"].map(label_map)
    out["degree"] = out["node"].map(dict(G.degree()))
    out["weighted_degree"] = out["node"].map(dict(G.degree(weight="weight")))
    out = out.sort_values(["k_shell", "degree", "weighted_degree"], ascending=False).reset_index(drop=True)

    top_shell = out[out["k_shell"] == kmax].copy()

    return {
        "platform": platform,
        "gdf_path": gdf_path,
        "xlsx_files": xlsx_files,
        "G": G,
        "core": core,
        "summary_df": out,
        "kmax": kmax,
        "top_shell_df": top_shell,
    }




def plot_k_shell_hist(core: Dict[int, int], title: str):
    vals = list(core.values())
    plt.figure(figsize=(8,4))
    plt.hist(vals, bins=range(min(vals), max(vals)+2))
    plt.title(title)
    plt.xlabel("k-shell index (core number)")
    plt.ylabel("Number of nodes")
    plt.show()


def plot_k_core_sizes(core: Dict[int,int], title: str):
    kmax = max(core.values()) if core else 0
    ks = list(range(1, kmax+1))
    sizes = []
    for k in ks:
        sizes.append(sum(1 for v in core.values() if v >= k))

    plt.figure(figsize=(8,4))
    plt.plot(ks, sizes, marker="o")
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("Size of k-core (
    plt.grid(True)
    plt.show()




def main(
    out_dir: Path,
    base_dir: Path | None = None,
    zip_path: Path | None = None,
    platforms: List[str] | None = None,
    make_plots: bool = True,
    do_content_analysis: bool = True,
) -> Dict[str, object]:
    """Run Q3 end-to-end and write outputs into out_dir."""
    cfg = Q03Config(
        base_dir=Path(base_dir) if base_dir is not None else Q03Config().base_dir,
        zip_path=Path(zip_path) if zip_path is not None else Q03Config().zip_path,
        platforms=tuple(platforms) if platforms is not None else Q03Config().platforms,
        figure_dpi=Q03Config().figure_dpi,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["figure.dpi"] = cfg.figure_dpi

    extracted = unzip_if_needed(cfg.zip_path, cfg.base_dir)
    if extracted:
        print(f"Unzipped {cfg.zip_path} into {cfg.base_dir.resolve()}")
    print("BASE_DIR =", cfg.base_dir.resolve())
    print("OUT_DIR  =", out_dir.resolve())

    platform_results: Dict[str, Dict[str, object]] = {}

    for p in cfg.platforms:
        res = run_kcore_for_platform(cfg.base_dir, p)
        platform_results[p] = res
        print(
            f"[{p}] nodes={res['G'].number_of_nodes():,} "
            f"edges={res['G'].number_of_edges():,}  "
            f"kmax={res['kmax']}  top_shell_nodes={len(res['top_shell_df']):,}"
        )


        (res["summary_df"]).to_csv(out_dir / f"{p}_core_summary.csv", index=False)
        (res["top_shell_df"]).to_csv(out_dir / f"{p}_top_shell.csv", index=False)

    if make_plots:
        for p, res in platform_results.items():
            plot_k_shell_hist(res["core"], f"{p}: k-shell histogram")
            plot_k_core_sizes(res["core"], f"{p}: k-core sizes")

    content_outputs = {}
    if do_content_analysis:
        for p, res in platform_results.items():
            xlsx_files = res["xlsx_files"]
            top_shell_users = [
                u for u in res["top_shell_df"]["username"].dropna().astype(str).tolist()
                if u and u != "nan"
            ]

            df_posts = load_platform_excels(xlsx_files)
            if df_posts.empty:
                print(f"[{p}] No Excel posts loaded -> skipping content analysis")
                continue


            df_posts = df_posts[df_posts["username"].isin(top_shell_users)].copy()
            print(
                f"[{p}] top-shell users in GDF: {len(top_shell_users)} "
                f"| posts found in excels: {len(df_posts):,}"
            )

            kw_df = tfidf_keywords_by_user(df_posts, top_shell_users, top_k=20)
            out_path = out_dir / f"{p}_top_shell_tfidf_keywords.csv"
            kw_df.to_csv(out_path, index=False)
            print(f"[{p}] saved: {out_path}")

            content_outputs[p] = kw_df

    return {
        "config": cfg,
        "platform_results": platform_results,
        "content_outputs": content_outputs,
        "out_dir": out_dir,
    }
