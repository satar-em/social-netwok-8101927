from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import random

import numpy as np




SEED: int = 42
random.seed(SEED)
np.random.seed(SEED)




PLATFORMS = ["twitter", "telegram", "instagram"]
GDF_NAME_FMT = "{platform}-10 to 24 dey.gdf"
XLSX_NAME_FMT = "{platform}-{day} dey.xlsx"

def project_root() -> Path:

    return Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Paths:
    data_zip: Path
    extract_dir: Path
    out_dir: Path

def resolve_paths(
    data_zip: str | os.PathLike | None = None,
    extract_dir: str | os.PathLike | None = None,
    out_dir: str | os.PathLike | None = None,
) -> Paths:
    root = project_root()

    data_zip_p = Path(data_zip) if data_zip is not None else (root / "data.zip")
    extract_dir_p = Path(extract_dir) if extract_dir is not None else (root / "data_extracted")
    out_dir_p = Path(out_dir) if out_dir is not None else (root / "outputs" / "q04")

    return Paths(
        data_zip=data_zip_p,
        extract_dir=extract_dir_p,
        out_dir=out_dir_p,
    )
