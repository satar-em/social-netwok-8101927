from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd

from .text_utils import normalize_fa

def unzip(zip_path: Path, extract_dir: Path) -> list[Path]:
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Cannot find {zip_path}. Put data.zip next to the notebook or pass zip_path=..."
        )
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    return sorted([p for p in extract_dir.rglob("*") if p.is_file()])

def parse_dey_day_from_filename(path: Path):

    m = re.search(r"-(\d+)\s*dey\.xlsx$", path.name, flags=re.I)
    return int(m.group(1)) if m else None

def load_all_xlsx(extract_dir: Path) -> pd.DataFrame:
    xlsx_paths = sorted(extract_dir.rglob("*.xlsx"))
    if len(xlsx_paths) == 0:
        raise RuntimeError("No .xlsx files found under extract_dir. Check the zip structure.")

    frames = []
    for p in xlsx_paths:
        day = parse_dey_day_from_filename(p)
        df = pd.read_excel(p)
        df.columns = [c.strip().lower() for c in df.columns]
        df["dey_day"] = day

        df["platform"] = df.get("platform", "unknown").astype(str).str.lower().str.strip()

        df["text_norm"] = df.get("text", "").astype(str).map(normalize_fa)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
