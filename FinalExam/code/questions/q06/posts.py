"""Post loading + activity features (ported from Q6 notebook)."""

from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import pytz
    from datetime import time
    TEHRAN_TZ = pytz.timezone("Asia/Tehran")
    UTC_TZ = pytz.UTC
except Exception:
    from zoneinfo import ZoneInfo
    from datetime import time
    TEHRAN_TZ = ZoneInfo("Asia/Tehran")
    UTC_TZ = ZoneInfo("UTC")


def load_platform_posts(platform: str, platform_dir: Path, cut_local_time: time) -> pd.DataFrame:
    files = sorted(platform_dir.glob("*.xlsx"))
    if len(files) != 5:
        raise ValueError(f"Expected 5 XLSX files for {platform}, found {len(files)}")

    all_df = []
    for fp in files:
        df = pd.read_excel(fp)
        df.columns = [c.strip().lower() for c in df.columns]

        if "date (jalali)" in df.columns and "date(jalali)" not in df.columns:
            df = df.rename(columns={"date (jalali)": "date(jalali)"})

        m = re.search(r"-(\d+)\s*dey", fp.name.lower())
        day = int(m.group(1)) if m else None
        df["dey_day"] = day

        df["username"] = df["username"].astype(str).str.strip()
        df["username_norm"] = df["username"].str.lower()

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date_utc"] = df["date"].dt.tz_localize(UTC_TZ, nonexistent="shift_forward", ambiguous="NaT")
        df["date_tehran"] = df["date_utc"].dt.tz_convert(TEHRAN_TZ)

        df["hour_tehran"] = df["date_tehran"].dt.hour + df["date_tehran"].dt.minute / 60.0

        df["after_cut"] = False
        if day == 17:
            df["after_cut"] = df["date_tehran"].dt.time >= cut_local_time

        all_df.append(df)

    posts = pd.concat(all_df, ignore_index=True)
    return posts


def build_activity_features(posts: pd.DataFrame) -> pd.DataFrame:
    g = posts.groupby("username_norm")

    feats = pd.DataFrame(
        {
            "posts_total": g.size(),
            "hour_mean": g["hour_tehran"].mean(),
            "hour_std": g["hour_tehran"].std(),
        }
    ).reset_index()

    day_counts = posts.pivot_table(
        index="username_norm", columns="dey_day", values="link", aggfunc="count", fill_value=0
    )
    day_counts = day_counts.add_prefix("posts_dey_").reset_index()

    feats = feats.merge(day_counts, on="username_norm", how="left").fillna(0)

    d17 = posts[posts["dey_day"] == 17].copy()
    if len(d17) > 0:
        d17_g = (
            d17.groupby("username_norm")["after_cut"]
            .agg(posts_dey17_total="size", posts_dey17_after_cut="sum")
            .reset_index()
        )
        d17_g["posts_dey17_before_cut"] = d17_g["posts_dey17_total"] - d17_g["posts_dey17_after_cut"]
        d17_g["share_dey17_after_cut"] = d17_g["posts_dey17_after_cut"] / d17_g["posts_dey17_total"].replace(0, np.nan)
        d17_g["share_dey17_after_cut"] = d17_g["share_dey17_after_cut"].fillna(0.0)
    else:
        d17_g = pd.DataFrame(
            columns=[
                "username_norm",
                "posts_dey17_total",
                "posts_dey17_after_cut",
                "posts_dey17_before_cut",
                "share_dey17_after_cut",
            ]
        )

    feats = feats.merge(d17_g, on="username_norm", how="left").fillna(0)

    feats["posts_pre_15_16"] = feats.get("posts_dey_15", 0) + feats.get("posts_dey_16", 0)
    feats["posts_post_18_19"] = feats.get("posts_dey_18", 0) + feats.get("posts_dey_19", 0)
    feats["post_pre_ratio"] = feats["posts_post_18_19"] / (feats["posts_pre_15_16"] + 1e-6)

    return feats
