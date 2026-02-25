from __future__ import annotations

import random
import numpy as np
import networkx as nx


def sample_triangles_and_balance(Gu: nx.Graph, faction: dict, num_samples: int = 20000, seed: int = 42):
    rnd = random.Random(seed)

    nodes_keep = [n for n in Gu.nodes() if faction.get(n, -1) >= 0]
    G = Gu.subgraph(nodes_keep).copy()

    nbrs = {u: list(G.neighbors(u)) for u in G.nodes()}
    nodes_list = list(G.nodes())
    if not nodes_list:
        return {"triangles": 0, "balanced": 0, "balance_ratio": np.nan}

    balanced = 0
    total = 0

    for _ in range(num_samples):
        u = rnd.choice(nodes_list)
        nu = nbrs[u]
        if len(nu) < 2:
            continue
        v, w = rnd.sample(nu, 2)
        if not G.has_edge(v, w):
            continue

        fu, fv, fw = faction[u], faction[v], faction[w]

        def s(a, b):
            return 1 if a == b else -1

        sign_prod = s(fu, fv) * s(fu, fw) * s(fv, fw)
        total += 1
        if sign_prod > 0:
            balanced += 1

    return {
        "triangles": total,
        "balanced": balanced,
        "balance_ratio": (balanced / total) if total > 0 else np.nan,
    }
