from __future__ import annotations

import os
import re
from glob import glob
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_platform_posts(extract_dir: str | Path, platform: str) -> pd.DataFrame:
    extract_dir = Path(extract_dir)
    files: List[str] = sorted(glob(str(extract_dir / platform / f"{platform}-* dey.xlsx")))
    if not files:
        raise FileNotFoundError(f"No xlsx files found for {platform} in {extract_dir/platform}")

    dfs = []
    for fp in files:
        df = pd.read_excel(fp)
        m = re.search(rf"{re.escape(platform)}-(\d+)\s+dey\.xlsx", os.path.basename(fp))
        day = int(m.group(1)) if m else None
        df["day_dey"] = day
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)


    all_df.columns = [c.strip().lower().replace(" ", "") for c in all_df.columns]
    return all_df


def load_all_posts(extract_dir: str | Path, platforms: List[str]) -> Dict[str, pd.DataFrame]:
    return {p: load_platform_posts(extract_dir, p) for p in platforms}
