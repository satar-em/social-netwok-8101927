"""Reporting helpers (ported from Q6 notebook)."""

from __future__ import annotations
import pandas as pd


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("final_label")
        .agg(
            n_accounts=("node_id", "count"),
            pr_sum=("pagerank", "sum"),
            pr_mean=("pagerank", "mean"),
            conf_mean=("final_conf", "mean"),
        )
        .sort_values("n_accounts", ascending=False)
    )
