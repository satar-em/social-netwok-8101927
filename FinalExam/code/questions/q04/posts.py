from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import XLSX_NAME_FMT
from .text_utils import simple_persian_normalize

def load_all_excels(extract_dir: Path, platform: str) -> pd.DataFrame:
    dfs = []
    for day in [15, 16, 17, 18, 19]:
        path = extract_dir / platform / XLSX_NAME_FMT.format(platform=platform, day=day)
        if not path.exists():
            continue
        df = pd.read_excel(path)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No xlsx files found for {platform} in days 15..19")

    out = pd.concat(dfs, ignore_index=True)

    if "username" not in out.columns or "text" not in out.columns:
        raise ValueError(f"Expected columns username/text not found. Available: {list(out.columns)}")

    out["username"] = out["username"].astype(str)
    out["text"] = out["text"].apply(simple_persian_normalize)
    return out
