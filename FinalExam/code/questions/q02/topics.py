"""
Optional: Role discovery via clustering (as in the notebook's optional section).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


def node_feature_table(G: nx.DiGraph, id_to_label: dict,
                       pagerank: dict, hubs: dict, auth: dict) -> pd.DataFrame:
    rows = []
    for n in G.nodes():
        rows.append({
            "node_id": n,
            "username": id_to_label.get(n),
            "in_deg": G.in_degree(n),
            "out_deg": G.out_degree(n),
            "in_w": G.in_degree(n, weight="weight"),
            "out_w": G.out_degree(n, weight="weight"),
            "pagerank": pagerank.get(n, 0.0),
            "hub": hubs.get(n, 0.0),
            "authority": auth.get(n, 0.0),
            "followers": pd.to_numeric(G.nodes[n].get("followers", np.nan), errors="coerce"),
            "posts": pd.to_numeric(G.nodes[n].get("posts", np.nan), errors="coerce"),
            "importance": pd.to_numeric(G.nodes[n].get("importance", np.nan), errors="coerce"),
        })
    df = pd.DataFrame(rows)

    for c in ["in_deg","out_deg","in_w","out_w","followers","posts"]:
        df[c] = np.log1p(df[c].fillna(0))

    return df


def cluster_roles(G: nx.DiGraph, id_to_label: dict, k: int = 4, random_state: int = 42):
    """
    Cluster nodes by structural + centrality + (optional) attribute features.

    Returns
    -------
    df_roles: pd.DataFrame
    summary: pd.DataFrame (cluster means)
    """

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    pr = nx.pagerank(G, weight="weight")
    hubs, auth = nx.hits(G, max_iter=1000, tol=1e-8, normalized=True)

    df = node_feature_table(G, id_to_label, pr, hubs, auth)

    feature_cols = ["in_deg","out_deg","in_w","out_w","pagerank","hub","authority","followers","posts","importance"]
    X = df[feature_cols].fillna(0).values
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    df["cluster"] = km.fit_predict(Xs)

    summary = df.groupby("cluster")[feature_cols].mean().sort_index()
    return df, summary
