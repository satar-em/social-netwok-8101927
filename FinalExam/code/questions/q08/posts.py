from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd

from .text_utils import normalize_fa

_XLSX_RE = re.compile(r"/(twitter|instagram|telegram)-(\d+)\s*dey\.xlsx$", flags=re.I)

_NUMERIC_COLS = ["engagement", "forward", "like", "comment", "impression", "copy", "follower"]

def load_posts_from_zip(data_zip: str | Path) -> pd.DataFrame:
    """Load all platform/day Excel files from `data.zip` into one DataFrame."""
    data_zip = Path(data_zip)
    frames: list[pd.DataFrame] = []

    with zipfile.ZipFile(data_zip, "r") as z:
        for name in z.namelist():
            if not name.lower().endswith(".xlsx"):
                continue
            m = _XLSX_RE.search(name)
            if not m:
                continue

            platform = m.group(1).lower()
            day = int(m.group(2))

            with z.open(name) as f:
                content = f.read()

            df = pd.read_excel(io.BytesIO(content))
            df.columns = [str(c).strip().lower() for c in df.columns]

            df["platform"] = platform
            df["dey_day"] = day

            if "text" not in df.columns:
                df["text"] = ""
            df["text_norm"] = df["text"].astype(str).map(normalize_fa)


            for col in _NUMERIC_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if "engagement" in df.columns:
                df["engagement_filled"] = df["engagement"].fillna(0)
            else:
                df["engagement_filled"] = 0

            frames.append(df)

    if not frames:
        raise FileNotFoundError("No matching Excel files found in the zip (expected *-(day) dey.xlsx).")

    posts = pd.concat(frames, ignore_index=True)
    return posts
