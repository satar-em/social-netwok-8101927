from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .io_gdf import read_gdf
from .text_utils import top_tokens


def topk_accounts_from_gdf(extract_dir: str | Path, platform: str, metrics: List[str], topk: int = 5) -> Dict[str, pd.DataFrame]:
    extract_dir = Path(extract_dir)
    gdf_path = extract_dir / platform / f"{platform}-10 to 24 dey.gdf"
    nodes, _edges = read_gdf(gdf_path)

    tables: Dict[str, pd.DataFrame] = {}
    for m in metrics:
        if m not in nodes.columns:

            continue
        t = nodes.sort_values(m, ascending=False).head(topk)[["name", "label", m]].copy()
        t.columns = ["id", "username", m]
        tables[m] = t
    return tables


def export_top_tables(out_dir: str | Path, platform: str, tables: Dict[str, pd.DataFrame]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for m, t in tables.items():
        t.to_csv(out_dir / f"top5_{platform}_{m}.csv", index=False)


def union_top_usernames(top_tables: Dict[str, Dict[str, pd.DataFrame]], platform: str) -> List[str]:
    acc = set()
    for _m, t in top_tables.get(platform, {}).items():
        acc.update(t["username"].astype(str).tolist())
    return sorted(acc)


def _pick_rank_metric(df: pd.DataFrame) -> Optional[str]:

    candidates = ["forward", "engagement", "impression", "like", "comment", "copy"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def account_summary(df: pd.DataFrame, username: str) -> Optional[Dict]:
    sub = df[df["username"].astype(str).str.lower() == username.lower()].copy()
    if sub.empty:
        return None


    for col in ["forward","engagement","impression","like","comment","copy","follower"]:
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")

    sentiment_counts = sub["sentiment"].value_counts(dropna=False).to_dict() if "sentiment" in sub.columns else {}
    tokens = top_tokens(sub.get("text", pd.Series([], dtype=str)).astype(str).tolist(), n=20)

    rank_metric = _pick_rank_metric(sub)
    if rank_metric is not None:
        top_posts = sub.sort_values(rank_metric, ascending=False).head(5).copy()
    else:
        top_posts = sub.head(5).copy()


    keep_cols = ["day_dey","date","name","username","link","text","forward","engagement","impression","like","comment","copy","sentiment"]
    cols = [c for c in keep_cols if c in top_posts.columns]
    if cols:
        top_posts = top_posts[cols]

    return {
        "n_posts": int(len(sub)),
        "sentiment_counts": sentiment_counts,
        "tokens": tokens,
        "top_posts": top_posts,
        "rank_metric": rank_metric,
    }


def build_summaries(posts_by_platform: Dict[str, pd.DataFrame], top_tables: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, Optional[Dict]]]:
    summaries: Dict[str, Dict[str, Optional[Dict]]] = {}
    for platform, df in posts_by_platform.items():
        accounts = union_top_usernames(top_tables, platform)
        summaries[platform] = {acc: account_summary(df, acc) for acc in accounts}
    return summaries


def export_top_posts(out_dir: str | Path, platform: str, summaries: Dict[str, Dict[str, Optional[Dict]]]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for acc, st in summaries.get(platform, {}).items():
        if not st:
            continue
        st["top_posts"].to_csv(out_dir / f"top_posts_{platform}_{acc}.csv", index=False)
