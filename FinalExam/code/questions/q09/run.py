from __future__ import annotations

from pathlib import Path
from collections import Counter
import random

import numpy as np
import pandas as pd

from .config import SEED, K_ANTI
from .data import ensure_extracted, discover_paths
from .graphs import load_gdfs, build_working_graph, attach_seed_to_graph
from .posts import load_posts
from .stance import build_seed_labels
from .propagation import label_propagation_binary, finalize_binary_labels
from .communities import anti_subfactions
from .factions import build_factions
from .balance import sample_triangles_and_balance
from .matrix import inter_faction_matrix
from .keywords import top_terms_for_faction


def run(
    data_zip: str | Path = "data.zip",
    extract_dir: str | Path = "data_extracted",
    output_dir: str | Path = "outputs/q09",
    platforms: tuple[str, ...] = ("twitter", "telegram", "instagram"),
    topk_anti: int = K_ANTI,
    lp_tau: float = 0.2,
    lp_max_iter: int = 60,
    triangle_samples: int = 30000,
    seed_thr: float = 0.4,
    seed_min_hits: int = 1,
    export_keywords: bool = True,
) -> dict:
    """Run the full Q9 pipeline and save outputs.

    Returns a small summary dict suitable for `print(q09.run())`.
    """
    random.seed(SEED)
    np.random.seed(SEED)

    extract_dir = ensure_extracted(data_zip=data_zip, extract_dir=extract_dir)
    gdf_paths, xlsx_globs = discover_paths(extract_dir)


    platforms = tuple(p for p in platforms if p in gdf_paths)

    nodes_dfs, edges_dfs = load_gdfs({p: gdf_paths[p] for p in platforms})

    posts = {p: load_posts(p, xlsx_globs[p]) for p in platforms}
    seed_tables = {p: build_seed_labels(posts[p], min_hits=seed_min_hits, thr=seed_thr) for p in platforms}

    graphs = {p: build_working_graph(p, nodes_dfs, edges_dfs) for p in platforms}
    for p in platforms:
        Gd, _ = graphs[p]
        attach_seed_to_graph(Gd, seed_tables[p])


    binary_results = {}
    for p in platforms:
        Gd, Gu = graphs[p]
        seed_on_node = {n: d.get("seed_label", 0) for n, d in Gd.nodes(data=True)}
        nodes_list, f = label_propagation_binary(Gu, seed_on_node, max_iter=lp_max_iter)
        lab, conf = finalize_binary_labels(nodes_list, f, tau=lp_tau)
        binary_results[p] = (lab, conf)


    anti_results = {}
    for p in platforms:
        _, Gu = graphs[p]
        bin_lab, _ = binary_results[p]
        node2anti, comms, _ = anti_subfactions(Gu, bin_lab, k_anti=topk_anti)
        anti_results[p] = (node2anti, comms)


    factions = {}
    for p in platforms:
        Gd, _ = graphs[p]
        bin_lab, bin_conf = binary_results[p]
        node2anti, _ = anti_results[p]
        factions[p] = build_factions(Gd, bin_lab, bin_conf, node2anti)

    balance_stats = {p: sample_triangles_and_balance(graphs[p][1], factions[p], num_samples=triangle_samples, seed=SEED) for p in platforms}
    matrices = {p: inter_faction_matrix(graphs[p][1], factions[p]) for p in platforms}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    pd.DataFrame([{"platform": p, **balance_stats[p]} for p in platforms]).to_csv(out_dir / "balance_stats.csv", index=False)


    for p in platforms:
        Gd, _ = graphs[p]
        rows = []
        for n, d in Gd.nodes(data=True):
            rows.append({
                "node_id": n,
                "handle": d.get("label", ""),
                "seed_label": d.get("seed_label", 0),
                "seed_score": d.get("seed_score", 0.0),
                "stance_score_lp": d.get("stance_score_lp", 0.0),
                "stance_bin": d.get("stance_bin", 0),
                "faction": d.get("faction", -1),
            })
        pd.DataFrame(rows).to_csv(out_dir / f"node_labels_{p}.csv.gz", index=False, compression="gzip")
        matrices[p].to_csv(out_dir / f"inter_faction_matrix_{p}.csv")


    if export_keywords:
        kw_rows = []
        for p in platforms:
            anti_ids = sorted({fid for fid in factions[p].values() if fid >= 1})
            for fid in anti_ids:
                info = top_terms_for_faction(p, fid, graphs=graphs, factions=factions, posts=posts, topn=25)
                kw_rows.append({
                    "platform": p,
                    "anti_faction": fid,
                    "users_with_text": info["n_users_with_text"],
                    "hints": info["hint_counts"],
                    "top_terms": [t for t, _ in info["top_terms"][:15]],
                })
        pd.DataFrame(kw_rows).to_json(out_dir / "anti_faction_keywords.json", orient="records", force_ascii=False, indent=2)


    summary = {
        "output_dir": str(out_dir.resolve()),
        "platforms": list(platforms),
        "stance_counts": {p: Counter(binary_results[p][0].values()) for p in platforms},
        "faction_counts": {p: Counter(factions[p].values()) for p in platforms},
        "balance_ratio": {p: balance_stats[p]["balance_ratio"] for p in platforms},
        "files": sorted([f.name for f in out_dir.iterdir()]),
    }
    return summary
