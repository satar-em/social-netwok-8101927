import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import  random

def generate_barcodes(b):
    return [format(i, f'0{b}b') for i in range(2**b)]

def matches_pattern(code, pattern):
    for c, p in zip(code, pattern):
        if p != 'X' and c != p:
            return False
    return True


def build_deterministic_scale_free(b):
    N = 2**b
    codes = generate_barcodes(b)
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(b+1):
        S_pattern = '0'*i + 'X'*(b-i)
        D_pattern = 'X'*i + '1'*(b-i)
        S_nodes = [n for n, code in enumerate(codes) if matches_pattern(code, S_pattern)]
        D_nodes = [n for n, code in enumerate(codes) if matches_pattern(code, D_pattern)]
        for u in S_nodes:
            for v in D_nodes:
                G.add_edge(u, v)
    return G

b = 10
G_det = build_deterministic_scale_free(b)
N_det = G_det.number_of_nodes()
M_det = G_det.number_of_edges()

print(f'N_det={N_det}, M_det={M_det}')

A = nx.to_scipy_sparse_array(G_det, nodelist=sorted(G_det.nodes()))

plt.figure(figsize=(6,6))
plt.spy(A, markersize=1)
plt.title("Deterministic genetic model adjacency matrix (b=10, N=1024)")
plt.xlabel("Destination node index")
plt.ylabel("Source node index")
plt.tight_layout()
plt.show()

in_degrees = np.array([d for _, d in G_det.in_degree()])
out_degrees = np.array([d for _, d in G_det.out_degree()])

def degree_distribution(degrees):
    unique, counts = np.unique(degrees, return_counts=True)
    Pk = counts / degrees.size
    return unique, Pk

def fit_power_law(k_vals, Pk):
    mask = (k_vals > 0) & (Pk > 0)
    kv = k_vals[mask]
    pv = Pk[mask]
    logk = np.log(kv)
    logp = np.log(pv)
    slope, intercept = np.polyfit(logk, logp, 1)
    return slope, intercept, kv, pv

k_in, P_in = degree_distribution(in_degrees)
k_out, P_out = degree_distribution(out_degrees)

slope_in, intercept_in, kv_in, pv_in = fit_power_law(k_in, P_in)
slope_out, intercept_out, kv_out, pv_out = fit_power_law(k_out, P_out)

def plot_deg_with_fit(k_vals, Pk, slope, intercept, title):
    mask = (k_vals > 0) & (Pk > 0)
    kv = k_vals[mask]
    pv = Pk[mask]
    logk = np.log(kv)
    logp = np.log(pv)
    xfit = np.linspace(logk.min(), logk.max(), 100)
    yfit = slope * xfit + intercept

    plt.figure()
    plt.loglog(kv, pv, 'o', label='data')
    plt.loglog(np.exp(xfit), np.exp(yfit), '-', label=f'fit slope={slope:.2f}')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_deg_with_fit(k_in, P_in, slope_in, intercept_in,
                  "In-degree distribution (deterministic model)")
plot_deg_with_fit(k_out, P_out, slope_out, intercept_out,
                  "Out-degree distribution (deterministic model)")

print(f"slope_in={slope_in}, slope_out={slope_out}")


def generate_random_pattern(b, x):
    positions = list(range(b))
    X_pos = set(random.sample(positions, x))
    pattern = []
    for i in range(b):
        if i in X_pos:
            pattern.append('X')
        else:
            pattern.append(random.choice(['0','1']))
    return ''.join(pattern)

def build_RG_graph(b, x, r):
    N = 2**b
    codes = generate_barcodes(b)
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for rule_idx in range(r):
        S_pattern = generate_random_pattern(b, x)
        D_pattern = generate_random_pattern(b, x)
        S_nodes = [n for n, code in enumerate(codes) if matches_pattern(code, S_pattern)]
        D_nodes = [n for n, code in enumerate(codes) if matches_pattern(code, D_pattern)]
        for u in S_nodes:
            for v in D_nodes:
                G.add_edge(u, v)
    return G

def compute_density(G):
    N = G.number_of_nodes()
    M = G.number_of_edges()
    return M / (N**2)

b = 10
x_values = list(range(1, 9))
r_values = np.unique(np.round(np.logspace(np.log10(2), np.log10(160), num=10)).astype(int))
r_values = sorted(r_values)

density_sim = np.zeros((len(x_values), len(r_values)))
density_theory = np.zeros_like(density_sim)

for ix, x in enumerate(x_values):
    pi = 2**(x - b)
    for ir, r in enumerate(r_values):
        rho_th = 1 - (1 - pi**2)**(2*r)
        density_theory[ix, ir] = rho_th
        G_rg = build_RG_graph(b, x, r)
        density_sim[ix, ir] = compute_density(G_rg)

print(f'density_sim={density_sim}\,\ndensity_theory={density_theory}')


X_grid, R_grid = np.meshgrid(r_values, x_values)

plt.figure(figsize=(7,5))
plt.imshow(density_sim, aspect='auto', origin='lower',cmap='jet',
           extent=[min(r_values), max(r_values), min(x_values), max(x_values)])
plt.colorbar(label='Density (simulated)')
plt.xlabel('r (number of rules)')
plt.ylabel('x (number of X wildcards)')
plt.title('Random Genetic model density (simulated)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
plt.imshow(density_theory, aspect='auto', origin='lower',cmap='jet',
           extent=[min(r_values), max(r_values), min(x_values), max(x_values)])
plt.colorbar(label='Density (theoretical)')
plt.xlabel('r (number of rules)')
plt.ylabel('x (number of X wildcards)')
plt.title('Random Genetic model density (theoretical formula)')
plt.tight_layout()
plt.show()

slice_x_values = [2, 4, 6, 8]
slice_indices = [x_values.index(v) for v in slice_x_values]

plt.figure()
for idx, x in zip(slice_indices, slice_x_values):
    plt.plot(r_values, density_sim[idx], marker='o', label=f'x={x} (sim)')
    plt.plot(r_values, density_theory[idx], linestyle='--', label=f'x={x} (th)')
plt.xscale('log')
plt.xlabel('r')
plt.ylabel('Density')
plt.title('Density vs r for selected x values')
plt.legend()
plt.tight_layout()
plt.show()



