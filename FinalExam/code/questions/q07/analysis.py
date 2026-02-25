from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from . import config
from .extract import unzip, load_all_xlsx
from .text_utils import is_yazd_specific

@dataclass
class Q07Result:
    out_dir: Path
    summary_csv: Path | None
    candidates_csv: Path | None
    n_all_rows: int
    n_yazd_rows: int
    n_yazd_specific_rows: int

def filter_yazd(all_df: pd.DataFrame) -> pd.DataFrame:
    include_pat = "|".join(map(lambda s: __import__("re").escape(s), config.PROVINCE_TERMS))
    mask_include = all_df["text_norm"].str.contains(include_pat, na=False)
    mask_exclude = all_df["text_norm"].str.contains(config.EXCLUDE_PAT, na=False)
    return all_df[mask_include & (~mask_exclude)].copy()

def add_engagement(yazd_df: pd.DataFrame) -> pd.DataFrame:

    for col in ["engagement", "forward", "like", "comment", "impression", "copy"]:
        if col in yazd_df.columns:
            yazd_df[col] = pd.to_numeric(yazd_df[col], errors="coerce")

    yazd_df["engagement_filled"] = yazd_df.get("engagement", 0).fillna(0)
    return yazd_df

def summarize(yazd_df: pd.DataFrame) -> pd.DataFrame:
    return (
        yazd_df.groupby(["platform", "dey_day"])
        .agg(
            n_posts=("link", "count") if "link" in yazd_df.columns else ("text_norm", "count"),
            n_yazd_specific=("yazd_specific", "sum"),
            max_engagement=("engagement_filled", "max"),
            mean_engagement=("engagement_filled", "mean"),
        )
        .reset_index()
        .sort_values(["dey_day", "platform"])
    )

def cluster_day(yazd_df: pd.DataFrame, day: int, seed: int, min_k=2, max_k=6):
    sub = yazd_df[(yazd_df["dey_day"] == day) & (yazd_df["yazd_specific"])].copy()


    sub = sub[sub["text_norm"].str.contains(config.EVENT_TERMS, na=False)]

    if len(sub) < 8:
        return None, None

    texts = sub["text_norm"].tolist()

    vectorizer = TfidfVectorizer(
        max_features=4000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
        min_df=1,
    )
    X = vectorizer.fit_transform(texts)


    k = int(np.clip(len(sub) // 15 + 2, min_k, max_k))

    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    labels = km.fit_predict(X)
    sub["cluster"] = labels

    terms = np.array(vectorizer.get_feature_names_out())
    centroids = km.cluster_centers_
    topk = 12

    rows = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        rep = sub.iloc[idx].sort_values("engagement_filled", ascending=False).head(1).iloc[0]
        top_terms = terms[np.argsort(centroids[c])[-topk:]][::-1].tolist()

        rows.append({
            "dey_day": day,
            "cluster": int(c),
            "n_posts": int(len(idx)),
            "top_terms": ", ".join(top_terms),
            "rep_platform": rep.get("platform", ""),
            "rep_username": rep.get("username", ""),
            "rep_link": rep.get("link", ""),
            "rep_engagement": float(rep.get("engagement_filled", 0)),
            "rep_text": rep.get("text_norm", ""),
        })

    clusters_df = pd.DataFrame(rows).sort_values(["n_posts", "rep_engagement"], ascending=False)
    return sub, clusters_df

def main(
    out_dir: Path,
    zip_path: Path,
    extract_dir: Path,
    days: Iterable[int] = (15, 16, 17, 18, 19),
    make_plots: bool = False,
) -> Q07Result:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    np.random.seed(config.SEED)


    unzip(zip_path=Path(zip_path), extract_dir=Path(extract_dir))


    all_df = load_all_xlsx(Path(extract_dir))


    yazd_df = filter_yazd(all_df)
    yazd_df["yazd_specific"] = yazd_df["text_norm"].map(is_yazd_specific)
    yazd_df = add_engagement(yazd_df)


    summary_df = summarize(yazd_df)
    summary_csv = out_dir / "summary_counts_by_day_platform.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")


    all_clusters = []
    for day in list(days):
        _, clusters_df = cluster_day(yazd_df, day=day, seed=config.SEED)
        if clusters_df is None:
            continue
        clusters_path = out_dir / f"event_clusters_day{day}.csv"
        clusters_df.to_csv(clusters_path, index=False, encoding="utf-8-sig")
        all_clusters.append(clusters_df)

    candidates_csv = None
    if all_clusters:
        candidates = pd.concat(all_clusters, ignore_index=True)
        candidates_csv = out_dir / "event_candidates_by_day.csv"
        candidates.to_csv(candidates_csv, index=False, encoding="utf-8-sig")


    if make_plots:
        import matplotlib.pyplot as plt
        daily_counts = (
            yazd_df[yazd_df["dey_day"].between(min(days), max(days))]
            .groupby("dey_day")
            .agg(total=("text_norm", "count"), yazd_specific=("yazd_specific", "sum"))
            .reset_index()
        )
        plt.figure()
        plt.plot(daily_counts["dey_day"], daily_counts["total"], marker="o", label="Yazd-related (incl. province)")
        plt.plot(daily_counts["dey_day"], daily_counts["yazd_specific"], marker="o", label="Yazd-specific (focused)")
        plt.xlabel("Dey day")
        plt.ylabel("
        plt.title("Yazd mentions in top posts (15–19 Dey)")
        plt.legend()
        plt.show()

    return Q07Result(
        out_dir=out_dir,
        summary_csv=summary_csv,
        candidates_csv=candidates_csv,
        n_all_rows=int(len(all_df)),
        n_yazd_rows=int(len(yazd_df)),
        n_yazd_specific_rows=int(yazd_df["yazd_specific"].sum()),
    )
