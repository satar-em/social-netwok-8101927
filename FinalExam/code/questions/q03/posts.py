from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

def load_platform_excels(xlsx_files: List[Path]) -> pd.DataFrame:
    dfs = []
    for fp in xlsx_files:
        try:
            df = pd.read_excel(fp)
            df["__source_file"] = fp.name
            dfs.append(df)
        except Exception as e:
            print("Failed to read:", fp, "error:", e)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)


    cols = {c.lower(): c for c in out.columns}
    if "username" in cols:
        out["username"] = out[cols["username"]].astype(str)
    if "text" in cols:
        out["text"] = out[cols["text"]].astype(str)
    return out
