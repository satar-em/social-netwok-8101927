from __future__ import annotations

import os
import pandas as pd


def read_day(data_root: str, platform: str, day: int) -> pd.DataFrame:
    """Read one daily Excel file for a given platform/day."""
    path = os.path.join(data_root, platform, f"{platform}-{day} dey.xlsx")
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]


    if "date(jalali)" in df.columns and "date_jalali" not in df.columns:
        df = df.rename(columns={"date(jalali)": "date_jalali"})
    if "date (jalali)" in df.columns and "date_jalali" not in df.columns:
        df = df.rename(columns={"date (jalali)": "date_jalali"})

    df["platform"] = platform
    df["day_dey"] = day
    return df


def load_all_posts(data_root: str, platforms: list[str], days: list[int]) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for p in platforms:
        for d in days:
            dfs.append(read_day(data_root, p, d))
    return pd.concat(dfs, ignore_index=True)
