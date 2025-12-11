import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import  random


def build_1d_lattice(N, k=2):
    if k != 2:
        G = nx.Graph()
        G.add_nodes_from(range(N))
        half = k // 2
        for i in range(N):
            for d in range(1, half+1):
                G.add_edge(i, (i+d) % N)
                G.add_edge(i, (i-d) % N)
        return G
    else:
        return nx.cycle_graph(N)

def build_2d_lattice(N):
    L = int(round(N ** 0.5))
    L = max(L, 2)
    G = nx.grid_2d_graph(L, L)
    mapping = {node: i for i, node in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping)

def build_3d_lattice(N):
    L = int(round(N ** (1/3)))
    L = max(L, 2)
    G = nx.grid_graph(dim=[L, L, L])
    mapping = {node: i for i, node in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping)

def build_random_network(N, avg_k=4):
    p = avg_k / (N - 1)
    return nx.gnp_random_graph(N, p)

def approximate_average_shortest_path(G, num_sources=40):
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        G = G.subgraph(largest).copy()
    nodes = list(G.nodes())
    n = len(nodes)
    num_sources = min(num_sources, n)
    sources = random.sample(nodes, num_sources)
    total = 0.0
    count = 0
    for s in sources:
        lengths = nx.single_source_shortest_path_length(G, s)
        for t, d in lengths.items():
            if t != s:
                total += d
                count += 1
    return total / count, n


N_values = np.round(np.logspace(np.log10(500), np.log10(5000), num=6)).astype(int)
results = {kind: [] for kind in ["1d","2d","3d","rn"]}

for N in N_values:
    print(f"N = {N}")
    G1 = build_1d_lattice(N)
    d1, size1 = approximate_average_shortest_path(G1, num_sources=40)
    results["1d"].append((size1, d1))

    G2 = build_2d_lattice(N)
    d2, size2 = approximate_average_shortest_path(G2, num_sources=40)
    results["2d"].append((size2, d2))

    G3 = build_3d_lattice(N)
    d3, size3 = approximate_average_shortest_path(G3, num_sources=40)
    results["3d"].append((size3, d3))

    G4 = build_random_network(N)
    d4, size4 = approximate_average_shortest_path(G4, num_sources=40)
    results["rn"].append((size4, d4))

print(f'results={results}')


def extract_xy(kind):
    Ns = np.array([n for (n, d) in results[kind]])
    ds = np.array([d for (n, d) in results[kind]])
    return Ns, ds

Ns_1d, d_1d = extract_xy("1d")
Ns_2d, d_2d = extract_xy("2d")
Ns_3d, d_3d = extract_xy("3d")
Ns_rn, d_rn = extract_xy("rn")

print(f'd_1d={d_1d}')
print(f'd_2d={d_2d}')
print(f'd_3d={d_3d}')
print(f'd_rn={d_rn}')

plt.figure()
plt.plot(Ns_1d, d_1d, marker='o', label='1D lattice')
plt.plot(Ns_2d, d_2d, marker='o', label='2D lattice')
plt.plot(Ns_3d, d_3d, marker='o', label='3D lattice')
plt.plot(Ns_rn, d_rn, marker='o', label='Random network')
plt.xlabel('N')
plt.ylabel('Average shortest path length <d>')
plt.legend()
plt.tight_layout()
plt.show()

slopes = {}
plt.figure()
for Ns, ds, label_key, label in [
    (Ns_1d, d_1d, "1d", "1D lattice"),
    (Ns_2d, d_2d, "2d", "2D lattice"),
    (Ns_3d, d_3d, "3d", "3D lattice"),
    (Ns_rn, d_rn, "rn", "Random network"),
]:
    logN = np.log(Ns)
    logd = np.log(ds)
    slope, intercept = np.polyfit(logN, logd, 1)
    slopes[label_key] = (slope, intercept)
    xfit = np.linspace(logN.min(), logN.max(), 100)
    yfit = slope * xfit + intercept
    plt.plot(logN, logd, 'o', label=f'{label} data')
    plt.plot(xfit, yfit, '-', label=f'{label} fit (slope={slope:.2f})')

plt.xlabel('log N')
plt.ylabel('log <d>')
plt.legend()
plt.tight_layout()
plt.show()

print(f'slopes={slopes}')

