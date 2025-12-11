import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def simulate_ER_giant_and_susceptibility(N, k_values, R=30, seed_base=0):
    k_values = np.array(k_values, dtype=float)
    S_means, S_stds = [], []
    s_means, s_stds = [], []
    for idx, k in enumerate(k_values):
        p = k / (N - 1)
        S_list, s_list = [], []
        for r in range(R):
            G = nx.fast_gnp_random_graph(N, p, seed=seed_base + idx*R + r)
            comp_sizes = [len(c) for c in nx.connected_components(G)]
            comp_sizes.sort(reverse=True)
            NG = comp_sizes[0]
            S = NG / N
            if len(comp_sizes) > 1:
                small = comp_sizes[1:]
                s_avg = sum(small) / len(small)
            else:
                s_avg = 0.0
            S_list.append(S)
            s_list.append(s_avg)
        S_means.append(np.mean(S_list))
        S_stds.append(np.std(S_list))
        s_means.append(np.mean(s_list))
        s_stds.append(np.std(s_list))
    return {
        "k": k_values,
        "S_mean": np.array(S_means),
        "S_std": np.array(S_stds),
        "s_mean": np.array(s_means),
        "s_std": np.array(s_stds),
    }


# Part (a): N=1000, variable <k>
N_a = 1000
k_values_a = []

k_values_a += list(np.arange(0.0, 0.8, 0.1))
k_values_a += list(np.arange(0.8, 1.2001, 0.02))
k_values_a += list(np.arange(1.2, 5.0001, 0.1))

k_values_a = sorted(set(round(k, 4) for k in k_values_a))

results_a = simulate_ER_giant_and_susceptibility(N_a, k_values_a, R=30, seed_base=100)

plt.figure()
plt.errorbar(results_a["k"], results_a["S_mean"], yerr=results_a["S_std"], fmt='o', markersize=3)
plt.xlabel("<k>")
plt.ylabel("S = N_G / N")
plt.title("Order parameter S vs <k> (N=1000)")
plt.tight_layout()
plt.show()

plt.figure()
plt.errorbar(results_a["k"], results_a["s_mean"], yerr=results_a["s_std"], fmt='o', markersize=3)
plt.xlabel("<k>")
plt.ylabel("<s> (average finite cluster size)")
plt.title("<s> vs <k> (N=1000)")
plt.tight_layout()
plt.show()


def simulate_ER_giant_only(N, k_values, R=20, seed_base=0):
    k_values = np.array(k_values, dtype=float)
    S_means, S_stds = [], []
    for idx, k in enumerate(k_values):
        p = k / (N - 1)
        S_list = []
        for r in range(R):
            G = nx.fast_gnp_random_graph(N, p, seed=seed_base + idx*R + r)
            comp_sizes = [len(c) for c in nx.connected_components(G)]
            NG = max(comp_sizes)
            S_list.append(NG / N)
        S_means.append(np.mean(S_list))
        S_stds.append(np.std(S_list))
    return {
        "k": k_values,
        "S_mean": np.array(S_means),
        "S_std": np.array(S_stds),
    }

k_values_c = sorted(set([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2,
                         1.4, 1.6, 2.0, 3.0, 4.0, 5.0]))

results_c_100   = simulate_ER_giant_only(100,   k_values_c, R=40, seed_base=200)
results_c_1000  = simulate_ER_giant_only(1000,  k_values_c, R=30, seed_base=400)
results_c_10000 = simulate_ER_giant_only(10000, k_values_c, R=15, seed_base=600)

plt.figure()
plt.errorbar(results_c_100["k"], results_c_100["S_mean"], fmt='o-', markersize=3, label='N=100')
plt.errorbar(results_c_1000["k"], results_c_1000["S_mean"], fmt='s-', markersize=3, label='N=1000')
plt.errorbar(results_c_10000["k"], results_c_10000["S_mean"], fmt='^-', markersize=3, label='N=10000')
plt.xlabel("<k>")
plt.ylabel("S")
plt.title("S vs <k> for different N")
plt.legend()
plt.tight_layout()
plt.show()


# Part (d): critical state at N=10000, <k>=1
N_d = 10000
k_crit = 1.0
p_crit = k_crit / (N_d - 1)
G_crit = nx.fast_gnp_random_graph(N_d, p_crit, seed=1234)

comp_sizes = np.array([len(c) for c in nx.connected_components(G_crit)], dtype=int)

print(f'sorted(set(comp_sizes))[-10:]={sorted(set(comp_sizes))[-10:]}')

sizes = comp_sizes
total_components = len(sizes)

s_min = 1
s_max = sizes.max()
num_bins = 25
bins = np.logspace(np.log10(s_min), np.log10(s_max), num=num_bins+1)
counts, edges = np.histogram(sizes, bins=bins)

bin_centers = np.sqrt(edges[:-1] * edges[1:])
bin_widths = edges[1:] - edges[:-1]
P_bin = counts / (total_components * bin_widths)

mask = (counts > 0) & (bin_centers >= 2) & (bin_centers <= 150)
log_s = np.log(bin_centers[mask])
log_P = np.log(P_bin[mask])

slope, intercept = np.polyfit(log_s, log_P, 1)
alpha_est = -slope

plt.figure()
plt.loglog(bin_centers, P_bin, 'o', label='binned data')
s_fit = np.logspace(np.log10(2), np.log10(150), 100)
P_fit = np.exp(intercept) * s_fit**slope
plt.loglog(s_fit, P_fit, '-', label=f'fit slope={slope:.2f} (alpha~{alpha_est:.2f})')
plt.xlabel("s (component size)")
plt.ylabel("P(s)")
plt.title(f"Component size distribution at criticality (N={N_d}, <k>=1)")
plt.legend()
plt.tight_layout()
plt.show()

print(f'slope={slope}, alpha_est={alpha_est}')

