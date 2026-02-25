from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_posts(platform: str, xlsx_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for fp in xlsx_paths:
        df = pd.read_excel(fp)
        df["__file"] = fp.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)


    if "username" in out.columns:
        out["user_key"] = out["username"].astype(str).str.strip().str.lower().str.lstrip("@")
    elif "name" in out.columns:
        out["user_key"] = out["name"].astype(str).str.strip().str.lower().str.lstrip("@")
    else:
        raise ValueError("No username/name column found in Excel files.")

    if "text" not in out.columns:
        raise ValueError("No text column found in Excel files.")

    out["text"] = out["text"].astype(str).fillna("")
    return out
