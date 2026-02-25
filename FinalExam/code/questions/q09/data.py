from __future__ import annotations

from pathlib import Path
import zipfile


def _repo_root() -> Path:

    return Path(__file__).resolve().parents[2]


def ensure_extracted(data_zip: str | Path = "data.zip", extract_dir: str | Path = "data_extracted") -> Path:
    """Ensure the dataset is extracted and return the extraction directory.

    The assignment notebooks expect a structure like:
      data_extracted/
        twitter/*.gdf, *.xlsx
        telegram/*.gdf, *.xlsx
        instagram/*.gdf, *.xlsx
    """
    data_zip = Path(data_zip)
    extract_dir = Path(extract_dir)


    if not data_zip.exists():
        cand = _repo_root() / data_zip
        if cand.exists():
            data_zip = cand
        else:
            cand2 = _repo_root().parent / data_zip
            if cand2.exists():
                data_zip = cand2

    if not data_zip.exists():
        raise FileNotFoundError(
            f"Cannot find dataset zip: {data_zip} (also checked repo root).\n"
            f"Tip: place 'data.zip' next to allQuestion.ipynb, or pass data_zip=... to run()."
        )

    if not extract_dir.is_absolute():

        extract_dir = _repo_root() / extract_dir

    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(data_zip, "r") as z:
            z.extractall(extract_dir)

    return extract_dir


def discover_paths(extract_dir: str | Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    extract_dir = Path(extract_dir)
    gdf_paths = {
        "twitter":   next((extract_dir / "twitter").glob("*.gdf")),
        "telegram":  next((extract_dir / "telegram").glob("*.gdf")),
        "instagram": next((extract_dir / "instagram").glob("*.gdf")),
    }
    xlsx_globs = {
        "twitter":   sorted((extract_dir / "twitter").glob("*.xlsx")),
        "telegram":  sorted((extract_dir / "telegram").glob("*.xlsx")),
        "instagram": sorted((extract_dir / "instagram").glob("*.xlsx")),
    }
    return gdf_paths, xlsx_globs
