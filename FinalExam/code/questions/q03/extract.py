from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List, Tuple

def unzip_if_needed(zip_path: Path | None, base_dir: Path) -> bool:
    """Unzip zip_path into base_dir if zip_path exists. Returns True if extracted."""
    if zip_path is None:
        return False
    zip_path = Path(zip_path)
    if not zip_path.exists():
        return False
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(base_dir)
    return True


def locate_platform_files(base_dir: Path, platform: str) -> Tuple[Path, List[Path]]:
    """Find the platform (gdf, xlsx_files) under base_dir by searching common patterns."""
    base_dir = Path(base_dir)


    gdf_candidates = list(base_dir.rglob(f"{platform}-10 to 24 dey.gdf"))
    if not gdf_candidates:

        gdf_candidates = list(base_dir.rglob(f"{platform}*.gdf"))
    if not gdf_candidates:
        raise FileNotFoundError(f"No GDF found for platform={platform} under {base_dir.resolve()}")
    gdf_path = sorted(gdf_candidates, key=lambda p: len(str(p)))[0]


    xlsx_files = sorted(base_dir.rglob(f"{platform}-*.xlsx"))
    return gdf_path, xlsx_files
