from __future__ import annotations

import networkx as nx

from .config import SEED, MIN_ANTI_SIZE


def anti_subfactions(Gu: nx.Graph, bin_label: dict, k_anti: int = 6):
    anti_nodes = [n for n, l in bin_label.items() if l == -1]
    Ganti = Gu.subgraph(anti_nodes).copy()


    if hasattr(nx.community, "louvain_communities"):
        comms = nx.community.louvain_communities(Ganti, weight="weight", seed=SEED)
    else:

        comms = nx.community.greedy_modularity_communities(Ganti, weight="weight")
    comms = sorted(comms, key=len, reverse=True)

    comms = [c for c in comms if len(c) >= MIN_ANTI_SIZE]
    comms = comms[:k_anti]

    node2anti = {}
    for i, c in enumerate(comms, start=1):
        for n in c:
            node2anti[n] = i
    return node2anti, comms, Ganti
