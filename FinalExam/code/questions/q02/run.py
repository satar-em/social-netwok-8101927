"""
Entry point for Homework Q2.

Public API:
    run(output_dir: str) -> dict
"""

from __future__ import annotations

import os
from typing import Dict, Any

from .config import DATA_ZIP, EXTRACT_DIR, PLATFORMS, GDF_SUFFIX
from .extract import extract_dataset
from .analysis import run_platform, save_outputs


def run(output_dir: str  = "outputs/q02") -> Dict[str, Any]:
    """
    Build graphs + compute top-10 metrics for all platforms and save CSV outputs.

    Parameters
    ----------
    output_dir:
        Where to save CSV files (e.g., outputs/q02)

    Returns
    -------
    platform_results: dict
        {
          platform: {
            "G": nx.DiGraph,
            "id_to_label": dict,
            "metrics": { ... dataframes ... }
          }
        }
    """

    extract_dataset(DATA_ZIP, EXTRACT_DIR, clean=True)


    platform_results = {}
    for plat in PLATFORMS:
        G, id_to_label, metrics = run_platform(EXTRACT_DIR, plat, GDF_SUFFIX, k=10)
        platform_results[plat] = {"G": G, "id_to_label": id_to_label, "metrics": metrics}


    os.makedirs(output_dir, exist_ok=True)
    save_outputs(platform_results, output_dir)

    return platform_results
