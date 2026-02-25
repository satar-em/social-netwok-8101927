from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from .config import Q5Config
from .gdf import discover_platform_gdfs, load_gdf_as_digraph
from .communities import (
    project_to_undirected_sum_weights,
    run_louvain,
    run_lpa_undirected,
    run_infomap_fast,
    _topk_communities,
    _best_jaccard_matches,
    save_node2comm,
    maybe_export_gexf,
)


def run_pipeline(
    *,
    output_dir: str | Path,
    data_zip: str | Path = "data.zip",
    extract_dir: str | Path = "data",
    config: Q5Config | None = None,
) -> dict:
    """
    Executes Q5 end-to-end pipeline:
    - Extracts data_zip to extract_dir if needed
    - Loads directed weighted graphs from GDF
    - Runs Louvain baseline + (Infomap OR LPA-undirected)
    - Exports CSVs (top-10 matching + node2comm maps + summary)

    Returns a dict with paths + summary DataFrame.
    """
    cfg = (config or Q5Config()).normalized()


    np.random.seed(int(cfg.seed))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_zip = Path(data_zip)
    extract_dir = Path(extract_dir)


    if data_zip.exists() and not extract_dir.exists():
        with zipfile.ZipFile(data_zip, "r") as z:
            z.extractall(extract_dir)


    platform2gdf = discover_platform_gdfs(extract_dir)

    graphs_dir: dict[str, nx.DiGraph] = {}
    for platform, path in platform2gdf.items():
        if cfg.platforms and platform not in cfg.platforms:
            continue
        t0 = time.perf_counter()
        Gd = load_gdf_as_digraph(path, weight_col="weight")
        t1 = time.perf_counter()
        graphs_dir[platform] = Gd


    assert (cfg.run_infomap ^ cfg.run_lpa), "Choose exactly one: run_infomap or run_lpa"

    summary = []

    for platform in cfg.platforms:
        if platform not in graphs_dir:
            continue
        Gd = graphs_dir[platform]


        t0 = time.perf_counter()
        louv = run_louvain(Gd, seed=int(cfg.seed))
        t1 = time.perf_counter()


        if cfg.run_infomap:
            b_name = f"infomap_{cfg.infomap_preset}"
            b_map, b_meta = run_infomap_fast(Gd, seed=int(cfg.seed), preset=str(cfg.infomap_preset))
        else:
            b_name = "lpa_und"
            Gu = project_to_undirected_sum_weights(Gd)
            t0b = time.perf_counter()
            b_map = run_lpa_undirected(Gu, seed=int(cfg.seed))
            t1b = time.perf_counter()
            b_meta = {"seconds": float(t1b - t0b)}


        top10_L = _topk_communities(louv, k=10)
        top10_B = _topk_communities(b_map, k=10)

        df_L2B = pd.DataFrame(_best_jaccard_matches(top10_L, top10_B))
        df_B2L = pd.DataFrame(_best_jaccard_matches(top10_B, top10_L))

        df_L2B.to_csv(out_dir / f"{platform}_top10_Louvain_to_{b_name}.csv", index=False)
        df_B2L.to_csv(out_dir / f"{platform}_top10_{b_name}_to_Louvain.csv", index=False)


        save_node2comm(louv, out_dir / f"{platform}_node2comm_louvain.csv.gz", colname="louvain_cid")
        save_node2comm(b_map, out_dir / f"{platform}_node2comm_{b_name}.csv.gz", colname=f"{b_name}_cid")


        if cfg.export_gexf:
            node_attrs = {"louvain_cid": louv, f"{b_name}_cid": b_map}
            maybe_export_gexf(Gd, node_attrs, out_dir / f"{platform}_annotated.gexf")

        summary.append({
            "platform": platform,
            "n": int(Gd.number_of_nodes()),
            "m": int(Gd.number_of_edges()),
            "methodB": b_name,
            "methodB_seconds": b_meta.get("seconds", None),
            "methodB_num_modules": b_meta.get("num_modules", None),
            "methodB_codelength": b_meta.get("codelength", None),
            "louvain_seconds": float(t1 - t0),
            "louvain_num_comms": int(len(set(louv.values()))),
        })

    summary_df = pd.DataFrame(summary)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "output_dir": str(out_dir),
        "summary_path": str(summary_path),
        "summary_df": summary_df,
        "platform2gdf": {k: str(v) for k, v in platform2gdf.items()},
    }
