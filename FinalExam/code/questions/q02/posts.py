"""
Optional: Read per-day Excel posts and sample content.
"""

from __future__ import annotations

import os
import pandas as pd


def load_all_daily_posts(extract_dir: str, platform: str) -> pd.DataFrame:
    folder = os.path.join(extract_dir, platform)
    xlsx_files = sorted([f for f in os.listdir(folder) if f.endswith(".xlsx")])
    dfs = []
    for xf in xlsx_files:
        path = os.path.join(folder, xf)
        df = pd.read_excel(path)
        df["__file"] = xf
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def sample_posts(extract_dir: str, platform: str, username: str, n: int = 10, sort_by: str = "Forward") -> pd.DataFrame:
    df = load_all_daily_posts(extract_dir, platform)
    sub = df[df["Username"].astype(str).str.lower() == str(username).lower()].copy()
    if sub.empty:
        return sub

    if sort_by in sub.columns:
        sub = sub.sort_values(sort_by, ascending=False)

    keep_cols = [c for c in ["__file","Platform","Name","Username","Date","text","Forward","Impression","Engagement","Like","Comment","Link"] if c in sub.columns]
    return sub[keep_cols].head(n)
