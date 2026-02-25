from __future__ import annotations

import numpy as np
import networkx as nx
import scipy.sparse as sp


def label_propagation_binary(Gu: nx.Graph, seed_labels: dict, max_iter: int = 50, tol: float = 1e-6):
    nodes = list(Gu.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    rows, cols, data = [], [], []
    for u, v, w in Gu.edges(data="weight", default=1.0):
        i, j = idx[u], idx[v]
        ww = float(w) if w is not None else 1.0
        rows += [i, j]
        cols += [j, i]
        data += [ww, ww]
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    deg = np.array(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    Dinv = sp.diags(1.0 / deg)

    y = np.zeros(n, dtype=float)
    mask = np.zeros(n, dtype=bool)
    for node, lab in seed_labels.items():
        if node in idx and lab in (-1, 1):
            y[idx[node]] = float(lab)
            mask[idx[node]] = True

    f = y.copy()
    for _ in range(max_iter):
        f_old = f.copy()
        f = (Dinv @ (A @ f))
        f[mask] = y[mask]
        delta = np.linalg.norm(f - f_old, ord=1) / max(1, n)
        if delta < tol:
            break

    return nodes, f


def finalize_binary_labels(nodes, f, tau: float = 0.2):
    lab, conf = {}, {}
    for node, val in zip(nodes, f):
        conf[node] = float(val)
        if abs(val) < tau:
            lab[node] = 0
        else:
            lab[node] = 1 if val > 0 else -1
    return lab, conf
