"""Dataset extraction for HW Q6."""

from __future__ import annotations

import zipfile
from pathlib import Path


def extract_dataset(data_zip: str, extract_dir: str) -> dict[str, Path]:
    """Extract `data_zip` into `extract_dir` (only if not already extracted).

    Returns a mapping: platform -> folder Path.
    """
    zip_path = Path(data_zip)
    out_dir = Path(extract_dir)

    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        if not zip_path.exists():
            raise FileNotFoundError(f"Cannot find dataset zip: {zip_path.resolve()}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)

    return {p: out_dir / p for p in ["twitter", "telegram", "instagram"]}
