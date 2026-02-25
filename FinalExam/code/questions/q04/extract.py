from __future__ import annotations

from pathlib import Path
import zipfile

def unzip_if_needed(zip_path: Path, extract_dir: Path) -> None:
    """
    Extracts zip into extract_dir if extract_dir is empty / missing.
    """
    if extract_dir.exists() and any(extract_dir.iterdir()):
        return

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path.resolve()}")

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
