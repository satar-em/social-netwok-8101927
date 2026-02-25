"""
Dataset extraction utilities.
"""

from __future__ import annotations

import os
import zipfile
import shutil


def extract_dataset(data_zip: str, extract_dir: str, clean: bool = True) -> None:
    """
    Extract `data_zip` into `extract_dir`.

    Parameters
    ----------
    data_zip:
        Path to data.zip
    extract_dir:
        Destination folder
    clean:
        If True, remove extract_dir first (fresh extract)
    """
    if not os.path.exists(data_zip):
        raise FileNotFoundError(f"Cannot find dataset zip: {data_zip}")

    if clean and os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(data_zip, "r") as z:
        z.extractall(extract_dir)
