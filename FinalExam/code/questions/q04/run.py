from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import networkx as nx

from .config import resolve_paths, PLATFORMS, GDF_NAME_FMT, SEED
from .extract import unzip_if_needed
from .analysis import (
    load_platform_graph,
    choose_resolution_for_instagram,
    run_louvain_and_filter,
    attach_community_id,
    export_for_gephi,
    top_accounts_tables,
)
from .topics import community_keywords_ml


def run(
    out_dir: str | Path | None = None,
    data_zip: str | Path | None = None,
    extract_dir: str | Path | None = None,
    run_ml: bool = True,
):
    """
    Runs Question 4 end-to-end and writes artifacts into outputs/q04 by default.

    Outputs (per platform):
      - *_all_nodes.csv
      - *_all_edges.csv
      - *_big_communities_summary.csv
      - *_with_louvain.gexf
      - *_louvain_id_<cid>.gexf  (subgraph per big community)
      - *_community_top_accounts.csv
      - ml_topics.json  (optional)
    """
    paths = resolve_paths(
        data_zip=str(data_zip) if data_zip is not None else None,
        extract_dir=str(extract_dir) if extract_dir is not None else None,
        out_dir=str(out_dir) if out_dir is not None else None,
    )

    paths.out_dir.mkdir(parents=True, exist_ok=True)
    unzip_if_needed(paths.data_zip, paths.extract_dir)


    graphs = {}
    for platform in PLATFORMS:
        G, nodes, edges, gdf_path = load_platform_graph(paths.extract_dir, platform, GDF_NAME_FMT)
        graphs[platform] = {"G": G, "nodes": nodes, "edges": edges, "gdf_path": gdf_path}
        nodes.to_csv(paths.out_dir / f"{platform}_all_nodes.csv", index=False)
        edges.to_csv(paths.out_dir / f"{platform}_all_edges.csv", index=False)


    results = {}
    for platform in PLATFORMS:
        G: nx.Graph = graphs[platform]["G"]

        res = 1.0
        if platform == "instagram":
            res = choose_resolution_for_instagram(G, seed=SEED)

        comms, big_summary, min_size = run_louvain_and_filter(G, resolution=res, seed=SEED)
        node2cid = attach_community_id(G, comms)

        summary_path = paths.out_dir / f"{platform}_big_communities_summary.csv"
        big_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")


        partition = {}
        for cid, nodeset in enumerate(comms):
            for n in nodeset:
                partition[n] = cid
        nx.set_node_attributes(G, partition, "community")

        gexf_path = paths.out_dir / f"{platform}_with_louvain.gexf"
        export_for_gephi(G, gexf_path)

        results[platform] = {
            "resolution": res,
            "min_size_5pct": min_size,
            "comms": comms,
            "big_summary": big_summary,
            "node2cid": node2cid,
            "paths": {"summary_csv": summary_path, "gexf": gexf_path},
        }


        for cid in big_summary["community_id"].tolist():
            cid = int(cid)
            nodes_set = set(comms[cid])
            H = G.subgraph(nodes_set).copy()
            export_for_gephi(H, paths.out_dir / f"{platform}_louvain_id_{cid}.gexf")


    for platform in PLATFORMS:
        big_summary = results[platform]["big_summary"]
        if len(big_summary) == 0:
            continue

        G = graphs[platform]["G"]
        comms = results[platform]["comms"]

        rows = []
        for _, r in big_summary.iterrows():
            cid = int(r["community_id"])
            comm_nodes = comms[cid]

            df_wdeg, df_pr = top_accounts_tables(G, comm_nodes, top_k=10)

            rows.append(
                {
                    "platform": platform,
                    "community_id": cid,
                    "size": int(r["size"]),
                    "percent": float(r["percent"]),
                    "top10_weighted_degree": df_wdeg.to_dict(orient="records"),
                    "top10_pagerank": df_pr.to_dict(orient="records"),
                }
            )

        out_csv = paths.out_dir / f"{platform}_community_top_accounts.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")


    ml_rows = []
    if run_ml:
        for platform in PLATFORMS:
            big = results[platform]["big_summary"]
            if len(big) == 0:
                continue

            nodes_df = graphs[platform]["nodes"]
            comms = results[platform]["comms"]

            for cid in big["community_id"].tolist():
                cid = int(cid)
                comm_nodes = comms[cid]
                out = community_keywords_ml(
                    platform=platform,
                    comm_nodes=comm_nodes,
                    nodes_df=nodes_df,
                    extract_dir=paths.extract_dir,
                    top_terms=15,
                    n_topics=3,
                    min_docs=50,
                )
                if out is not None:
                    ml_rows.append({"platform": platform, "community_id": cid, **out})

        ml_path = paths.out_dir / "ml_topics.json"
        with open(ml_path, "w", encoding="utf-8") as f:
            json.dump(ml_rows, f, ensure_ascii=False, indent=2)

    return {"out_dir": paths.out_dir, "results": results}
