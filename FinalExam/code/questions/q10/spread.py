from __future__ import annotations

import numpy as np
import pandas as pd


def compute_spread_score(df: pd.DataFrame) -> pd.Series:
    """
    Spread proxy:
      - use forward if it's informative (>0)
      - otherwise fallback to engagement
    """
    forward = df.get("forward", pd.Series([0]*len(df))).fillna(0)
    engagement = df.get("engagement", pd.Series([0]*len(df))).fillna(0)

    is_forward_informative = forward > 0
    score = np.where(is_forward_informative, forward, engagement)
    return pd.Series(score, index=df.index)


def add_spread_columns(all_posts: pd.DataFrame) -> pd.DataFrame:
    df = all_posts.copy()
    df["spread_score"] = compute_spread_score(df)


    def _z(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            std = 1.0
        return (s - s.mean()) / std

    df["spread_z"] = df.groupby(["platform", "day_dey"])["spread_score"].transform(_z)
    return df


def topk_by_platform_day(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    return (
        df.sort_values(["platform", "day_dey", "spread_score"], ascending=[True, True, False])
          .groupby(["platform", "day_dey"])
          .head(k)
          .reset_index(drop=True)
    )


def topk_overall_by_day(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    return (
        df.sort_values(["day_dey", "spread_z", "spread_score"], ascending=[True, False, False])
          .groupby("day_dey")
          .head(k)
          .reset_index(drop=True)
    )
