from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import zipfile


@dataclass(frozen=True)
class Q10Config:
    data_zip: Path
    extract_dir: Path
    output_dir: Path
    platforms: tuple[str, ...] = ("twitter", "telegram", "instagram")
    days: tuple[int, ...] = (15, 16, 17, 18, 19)
    topk: int = 5
    seed: int = 42


def ensure_extracted(data_zip: Path, extract_dir: Path) -> None:
    """Extract the homework zip (if needed) into extract_dir."""
    data_zip = Path(data_zip)
    extract_dir = Path(extract_dir)

    if not data_zip.exists():
        raise FileNotFoundError(f"Cannot find data zip: {data_zip.resolve()}")

    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(data_zip, "r") as z:
            z.extractall(extract_dir)

    if not extract_dir.exists():
        raise FileNotFoundError(f"Expected extraction dir to exist: {extract_dir.resolve()}")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)


def pandas_defaults() -> None:
    pd.set_option("display.max_colwidth", 120)
