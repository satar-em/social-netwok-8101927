from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Iterable


def ensure_extracted(data_zip: str | Path, extract_dir: str | Path, platforms: Iterable[str]) -> Path:
    """Extract zip if extract_dir does not exist (or is empty)."""
    data_zip = Path(data_zip)
    extract_dir = Path(extract_dir)

    if not data_zip.exists():
        raise FileNotFoundError(
            f"data zip not found: {data_zip}\n"
            "Place data.zip at project root, or pass data_zip=... to q01.run()."
        )

    needs_extract = (not extract_dir.exists()) or (extract_dir.exists() and not any(extract_dir.iterdir()))
    if needs_extract:
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(data_zip) as zf:
            zf.extractall(extract_dir)


    for p in platforms:
        pdir = extract_dir / p
        if not pdir.exists():
            raise FileNotFoundError(f"Missing folder after extraction: {pdir}")

        _ = os.listdir(pdir)

    return extract_dir
