from __future__ import annotations

def main(
    out_dir="outputs/q11",
    zip_path="data.zip",
    data_dir="data",
    make_plots=True,
):
    """Run Q11 pipeline (ported from notebook) using pure .py execution."""
    from pathlib import Path

    OUT_DIR = Path(out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    DATA_DIR = Path(data_dir)
    DATA_ZIP = Path(zip_path) if zip_path else None


    if not make_plots:
        try:
            import matplotlib.pyplot as plt
            plt.show = lambda *a, **k: None
        except Exception:
            pass



    import os, re, math, json, zipfile, hashlib
    from pathlib import Path

    import numpy as np
    import pandas as pd

    import networkx as nx
    import matplotlib.pyplot as plt

    SEED = 42
    np.random.seed(SEED)


    print("Python:", __import__("sys").version)
    print("pandas:", pd.__version__)
    print("networkx:", nx.__version__)



    if DATA_ZIP.exists() and not DATA_DIR.exists():
        with zipfile.ZipFile(DATA_ZIP, "r") as z:
            z.extractall(DATA_DIR)
        print("Extracted to:", DATA_DIR.resolve())
    else:
        print("DATA_DIR exists:", DATA_DIR.exists(), "| Using:", DATA_DIR.resolve())


    def normalize_text(s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.strip()

        s = re.sub(r"https?://\S+|www\.\S+", " ", s)

        s = re.sub(r"^\s*RT\s+@\w+:\s*", " ", s, flags=re.IGNORECASE)

        s = re.sub(r"[^\w\u0600-\u06FF
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    def tokenize(s: str):
        s = normalize_text(s)
        if not s:
            return []
        toks = s.split()

        toks = [t for t in toks if (len(t) >= 2 or t.startswith("
        return toks


    def _token_hash64(token: str) -> int:
        h = hashlib.md5(token.encode("utf-8", errors="ignore")).digest()
        return int.from_bytes(h[:8], "big", signed=False)

    def simhash64(text: str) -> int:
        toks = tokenize(text)
        if not toks:
            return 0
        freq = {}
        for t in toks:
            freq[t] = freq.get(t, 0) + 1

        v = [0] * 64
        for t, w in freq.items():
            h = _token_hash64(t)
            for i in range(64):
                bit = (h >> i) & 1
                v[i] += w if bit else -w

        fp = 0
        for i in range(64):
            if v[i] >= 0:
                fp |= (1 << i)
        return fp

    def hamming64(a: int, b: int) -> int:
        return (a ^ b).bit_count()


    def simhash_bands(fp: int, bands=4, band_bits=16):
        masks = (1 << band_bits) - 1
        out = []
        for b in range(bands):
            shift = b * band_bits
            out.append((b, (fp >> shift) & masks))
        return out

    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0]*n
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        def union(self, a, b):
            ra, rb = self.find(a), self.find(b)
            if ra == rb:
                return
            if self.rank[ra] < self.rank[rb]:
                self.parent[ra] = rb
            elif self.rank[ra] > self.rank[rb]:
                self.parent[rb] = ra
            else:
                self.parent[rb] = ra
                self.rank[ra] += 1


    def load_gdf_digraph(gdf_path):
        gdf_path = Path(gdf_path)
        lines = gdf_path.read_text(encoding="utf-8", errors="ignore").splitlines()

        node_i = None
        edge_i = None
        for i, line in enumerate(lines):
            if line.lower().startswith("nodedef>"):
                node_i = i
            if line.lower().startswith("edgedef>"):
                edge_i = i
                break
        if node_i is None or edge_i is None:
            raise ValueError(f"Could not find nodedef/edgedef in {gdf_path}")

        node_header = lines[node_i].split(">")[1].strip()
        node_cols = [c.split()[0] for c in node_header.split(",")]

        nodes = []
        for line in lines[node_i+1:edge_i]:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            row = dict(zip(node_cols, parts))
            nodes.append(row)

        edge_header = lines[edge_i].split(">")[1].strip()
        edge_cols = [c.split()[0] for c in edge_header.split(",")]

        edges = []
        for line in lines[edge_i+1:]:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            row = dict(zip(edge_cols, parts))
            edges.append(row)

        G = nx.DiGraph()
        node_id_col = "name" if "name" in node_cols else node_cols[0]
        for r in nodes:
            nid = r.get(node_id_col)
            if nid is None:
                continue
            G.add_node(str(nid), **r)

        src_col = "node1" if "node1" in edge_cols else ("source" if "source" in edge_cols else edge_cols[0])
        dst_col = "node2" if "node2" in edge_cols else ("target" if "target" in edge_cols else edge_cols[1])
        w_col = "weight" if "weight" in edge_cols else None

        for r in edges:
            u = str(r.get(src_col))
            v = str(r.get(dst_col))
            if not u or not v:
                continue
            w = float(r.get(w_col, 1.0)) if w_col else 1.0
            if G.has_edge(u, v):
                G[u][v]["weight"] += w
            else:
                G.add_edge(u, v, weight=w)
        return G

    def pagerank_weighted(G, alpha=0.85):
        return nx.pagerank(G, alpha=alpha, weight="weight")


    platforms = ["twitter", "telegram", "instagram"]
    graphs = {}
    pr = {}

    for p in platforms:
        gdf = DATA_DIR / p / f"{p}-10 to 24 dey.gdf"
        G = load_gdf_digraph(gdf)
        graphs[p] = G
        pr[p] = pagerank_weighted(G, alpha=0.85)
        print(p, "| nodes:", G.number_of_nodes(), "| edges:", G.number_of_edges(), "| directed:", G.is_directed())


    pr_df = {}
    for p in platforms:
        G = graphs[p]
        rows = []
        for node_id, val in pr[p].items():
            label = G.nodes[node_id].get("label", "")
            rows.append({
                "node_id": str(node_id),
                "label": str(label),
                "label_norm": str(label).strip().lower(),
                "pagerank": float(val),
            })
        pr_df[p] = (
            pd.DataFrame(rows)
            .sort_values("pagerank", ascending=False)
            .reset_index(drop=True)
        )

    pr_df["twitter"].head()


    DAYS = [15, 16, 17, 18, 19]

    def load_platform_posts(platform: str) -> pd.DataFrame:
        frames = []
        for d in DAYS:
            path = DATA_DIR / platform / f"{platform}-{d} dey.xlsx"
            df = pd.read_excel(path)
            df["day_dey"] = d
            frames.append(df)
        out = pd.concat(frames, ignore_index=True)

        out["username_norm"] = out["username"].astype(str).str.strip().str.lower()

        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")

        out["text_norm"] = out["text"].astype(str).map(normalize_text)

        out["forward"] = pd.to_numeric(out.get("forward", 0), errors="coerce").fillna(0)
        out["engagement"] = pd.to_numeric(out.get("engagement", 0), errors="coerce").fillna(0)
        out["spread_score"] = np.where(out["forward"] > 0, out["forward"], out["engagement"])

        return out

    posts = {p: load_platform_posts(p) for p in platforms}
    for p in platforms:
        print(p, posts[p].shape, "unique users:", posts[p]["username_norm"].nunique())


    K_MIN_ACCOUNTS = 5
    HAMMING_MAX = 3
    MAX_BUCKET_SIZE = 300

    def cluster_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["simhash64"] = df["text_norm"].map(simhash64).astype(np.uint64)

        n = len(df)
        uf = UnionFind(n)


        for _, idxs in df.groupby("text_norm").indices.items():
            if len(idxs) > 1:
                base = idxs[0]
                for j in idxs[1:]:
                    uf.union(base, j)


        buckets = {}
        fps = df["simhash64"].tolist()

        for i, fp in enumerate(fps):
            for band_key in simhash_bands(int(fp)):
                buckets.setdefault(band_key, []).append(i)

        for _, idxs in buckets.items():
            if len(idxs) < 2:
                continue
            if len(idxs) > MAX_BUCKET_SIZE:
                continue
            for a_i in range(len(idxs)):
                ia = idxs[a_i]
                for b_i in range(a_i + 1, len(idxs)):
                    ib = idxs[b_i]
                    if uf.find(ia) == uf.find(ib):
                        continue
                    if hamming64(int(fps[ia]), int(fps[ib])) <= HAMMING_MAX:
                        uf.union(ia, ib)

        roots = [uf.find(i) for i in range(n)]
        df["cluster_root"] = roots
        root_map = {}
        for r in sorted(set(roots)):
            root_map[r] = f"c{len(root_map):05d}"
        df["cluster_id"] = df["cluster_root"].map(root_map)

        acc_counts = df.groupby("cluster_id")["username_norm"].nunique()
        df["cluster_unique_accounts"] = df["cluster_id"].map(acc_counts).astype(int)
        df["is_coordinated"] = df["cluster_unique_accounts"] >= K_MIN_ACCOUNTS
        return df


    clustered = []
    for p in platforms:
        for d in DAYS:
            sub = posts[p][posts[p]["day_dey"] == d].copy()
            sub["platform"] = p
            sub = cluster_near_duplicates(sub)
            clustered.append(sub)

    clustered = pd.concat(clustered, ignore_index=True)

    clustered[["platform","day_dey","username_norm","spread_score","cluster_id","cluster_unique_accounts","is_coordinated"]].head()


    def platform_day_metrics(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (p, d), sub in df.groupby(["platform", "day_dey"]):
            total_posts = len(sub)
            total_spread = float(sub["spread_score"].sum())

            coord = sub[sub["is_coordinated"]]
            coord_posts = len(coord)
            coord_spread = float(coord["spread_score"].sum())

            coord_clusters = coord.groupby("cluster_id")["username_norm"].nunique()
            n_coord_clusters = int(coord_clusters.shape[0])
            avg_accounts = float(coord_clusters.mean()) if n_coord_clusters else 0.0

            rows.append({
                "platform": p,
                "day_dey": d,
                "total_posts": total_posts,
                "coord_posts": coord_posts,
                "coord_post_frac": coord_posts / total_posts if total_posts else 0.0,
                "total_spread": total_spread,
                "coord_spread": coord_spread,
                "coord_spread_frac": coord_spread / total_spread if total_spread else 0.0,
                "n_coord_clusters": n_coord_clusters,
                "avg_unique_accounts_per_coord_cluster": avg_accounts,
            })
        return pd.DataFrame(rows).sort_values(["platform","day_dey"]).reset_index(drop=True)

    metrics = platform_day_metrics(clustered)
    metrics.to_csv(OUT_DIR/"coordination_metrics_platform_day.csv", index=False)
    metrics


    def top_clusters(df: pd.DataFrame, topn=15) -> pd.DataFrame:
        '''
        Return top coordinated clusters per (platform, day).
        Representative text/link are taken from the highest-spread post inside the cluster.
        '''
        sub = df[df["is_coordinated"]].copy()
        if sub.empty:
            return pd.DataFrame(columns=[
                "platform","day_dey","cluster_id","posts_in_cluster","unique_accounts",
                "total_spread","rep_text","rep_link"
            ])

        def summarize_cluster(g: pd.DataFrame) -> pd.Series:
            g2 = g.sort_values("spread_score", ascending=False)
            return pd.Series({
                "posts_in_cluster": len(g2),
                "unique_accounts": g2["username_norm"].nunique(),
                "total_spread": float(g2["spread_score"].sum()),
                "rep_text": g2.iloc[0]["text"],
                "rep_link": g2.iloc[0]["link"] if "link" in g2.columns else "",
            })

        agg = (
            sub.groupby(["platform","day_dey","cluster_id"], as_index=False)
               .apply(summarize_cluster)
               .reset_index(drop=True)
        )

        agg = agg.sort_values(
            ["platform","day_dey","unique_accounts","total_spread"],
            ascending=[True,True,False,False],
        )

        out = agg.groupby(["platform","day_dey"]).head(topn).reset_index(drop=True)
        return out

    topc = top_clusters(clustered, topn=15)
    topc.to_csv(OUT_DIR/"top_coordinated_clusters_platform_day.csv", index=False)
    topc.head(10)


    def account_coordination(df: pd.DataFrame) -> pd.DataFrame:
        coord = df[df["is_coordinated"]].copy()

        clusters_per_acc = coord.groupby(["platform","username_norm"])["cluster_id"].nunique().rename("n_coord_clusters_participated")
        posts_per_acc = coord.groupby(["platform","username_norm"]).size().rename("coord_posts")
        spread_per_acc = coord.groupby(["platform","username_norm"])["spread_score"].sum().rename("coord_spread")

        total_posts = df.groupby(["platform","username_norm"]).size().rename("total_posts")
        total_spread = df.groupby(["platform","username_norm"])["spread_score"].sum().rename("total_spread")

        out = pd.concat([clusters_per_acc, posts_per_acc, spread_per_acc, total_posts, total_spread], axis=1).fillna(0).reset_index()
        out["coord_post_frac_user"] = out["coord_posts"] / out["total_posts"].replace(0, np.nan)
        out["coord_spread_frac_user"] = out["coord_spread"] / out["total_spread"].replace(0, np.nan)
        out = out.fillna(0)


        pr_all = []
        for p in platforms:
            t = pr_df[p][["label_norm","pagerank"]].copy()
            t["platform"] = p
            t = t.rename(columns={"label_norm":"username_norm"})
            pr_all.append(t[["platform","username_norm","pagerank"]])

        pr_all = pd.concat(pr_all, ignore_index=True).drop_duplicates(["platform","username_norm"])
        out = out.merge(pr_all, on=["platform","username_norm"], how="left")
        out["pagerank"] = out["pagerank"].fillna(0.0)
        return out

    acc_scores = account_coordination(clustered)
    acc_scores.to_csv(OUT_DIR/"account_coordination_scores.csv", index=False)

    acc_scores.sort_values(["platform","n_coord_clusters_participated","pagerank"], ascending=[True,False,False]).head(15)



    plt.figure()
    for p in platforms:
        sub = metrics[metrics["platform"]==p]
        plt.plot(sub["day_dey"], sub["coord_post_frac"], marker="o", label=p)
    plt.xlabel("Dey day")
    plt.ylabel("Fraction of posts in coordinated clusters")
    plt.title("Coordination intensity by day (posts fraction)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR/"coord_intensity_posts_frac.png", dpi=200)
    plt.show()


    plt.figure()
    for p in platforms:
        sub = metrics[metrics["platform"]==p]
        plt.plot(sub["day_dey"], sub["coord_spread_frac"], marker="o", label=p)
    plt.xlabel("Dey day")
    plt.ylabel("Fraction of spread score in coordinated clusters")
    plt.title("Coordination intensity by day (spread fraction)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR/"coord_intensity_spread_frac.png", dpi=200)
    plt.show()



    plt.figure()
    for p in platforms:
        sub = acc_scores[acc_scores["platform"]==p].copy()
        x = np.log10(sub["pagerank"].values + 1e-12)
        y = sub["n_coord_clusters_participated"].values
        plt.scatter(x, y, alpha=0.35, label=p, s=18)
    plt.xlabel("log10(PageRank + 1e-12)")
    plt.ylabel("
    plt.title("Influence vs coordination participation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR/"pagerank_vs_coord_clusters.png", dpi=200)
    plt.show()


    def cross_platform_sync(top_clusters_df: pd.DataFrame, hmax=4) -> pd.DataFrame:
        items = top_clusters_df.copy()
        items["rep_text_norm"] = items["rep_text"].astype(str).map(normalize_text)
        items["fp"] = items["rep_text_norm"].map(simhash64).astype(np.uint64)

        rows = []
        for d in DAYS:
            day_items = items[items["day_dey"]==d].reset_index(drop=True)
            if day_items.empty:
                continue

            buckets = {}
            fps = day_items["fp"].tolist()
            for i, fp in enumerate(fps):
                for band_key in simhash_bands(int(fp)):
                    buckets.setdefault(band_key, []).append(i)

            uf = UnionFind(len(day_items))
            for _, idxs in buckets.items():
                if len(idxs) < 2 or len(idxs) > 200:
                    continue
                for a_i in range(len(idxs)):
                    ia = idxs[a_i]
                    for b_i in range(a_i+1, len(idxs)):
                        ib = idxs[b_i]
                        if day_items.loc[ia, "platform"] == day_items.loc[ib, "platform"]:
                            continue
                        if hamming64(int(fps[ia]), int(fps[ib])) <= hmax:
                            uf.union(ia, ib)

            roots = [uf.find(i) for i in range(len(day_items))]
            day_items["sync_group"] = roots

            for g, sub in day_items.groupby("sync_group"):
                plats = sorted(sub["platform"].unique().tolist())
                if len(plats) < 2:
                    continue
                rows.append({
                    "day_dey": d,
                    "platforms": ",".join(plats),
                    "n_platforms": len(plats),
                    "n_clusters": len(sub),
                    "total_unique_accounts": int(sub["unique_accounts"].sum()),
                    "total_spread": float(sub["total_spread"].sum()),
                    "rep_text": sub.sort_values("total_spread", ascending=False).iloc[0]["rep_text"],
                })

        if not rows:
            return pd.DataFrame(columns=[
                "day_dey","platforms","n_platforms","n_clusters",
                "total_unique_accounts","total_spread","rep_text"
            ])

        out = pd.DataFrame(rows).sort_values(["day_dey","n_platforms","total_spread"], ascending=[True,False,False])
        return out

    sync = cross_platform_sync(topc, hmax=4)
    sync.to_csv(OUT_DIR/"cross_platform_synchronization.csv", index=False)
    sync.head(20)

    return {
        "out_dir": str(OUT_DIR.resolve()),
        "data_dir": str(DATA_DIR.resolve()),
    }
