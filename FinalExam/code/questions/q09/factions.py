from __future__ import annotations

import networkx as nx


def build_factions(Gd: nx.DiGraph, bin_lab: dict, bin_conf: dict, node2anti: dict):
    faction = {}
    for n in Gd.nodes():
        if bin_lab.get(n, 0) == 1:
            faction[n] = 0
        elif bin_lab.get(n, 0) == -1:
            faction[n] = node2anti.get(n, -1)
        else:
            faction[n] = -1

    nx.set_node_attributes(Gd, faction, "faction")
    nx.set_node_attributes(Gd, bin_lab, "stance_bin")
    nx.set_node_attributes(Gd, bin_conf, "stance_score_lp")
    return faction
