from __future__ import annotations

from collections import defaultdict
import pandas as pd
import networkx as nx


def inter_faction_matrix(Gu: nx.Graph, faction: dict) -> pd.DataFrame:
    labeled = [n for n in Gu.nodes() if faction.get(n, -1) >= 0]
    G = Gu.subgraph(labeled)

    pairs = defaultdict(float)
    facs = set()
    for u, v, w in G.edges(data="weight", default=1.0):
        fu, fv = faction[u], faction[v]
        facs.add(fu)
        facs.add(fv)
        a, b = (fu, fv) if fu <= fv else (fv, fu)
        pairs[(a, b)] += float(w) if w is not None else 1.0

    facs = sorted(facs)
    mat = pd.DataFrame(0.0, index=facs, columns=facs)
    for (a, b), ww in pairs.items():
        mat.loc[a, b] += ww
        if a != b:
            mat.loc[b, a] += ww
    return mat
