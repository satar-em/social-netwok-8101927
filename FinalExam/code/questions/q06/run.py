"""Entry point for HW Q6.

Goal: make it runnable with **no args**:

    import questions.q06 as q06
    q06.run()

Outputs go under outputs/q06/<platform>/...

Notes on reload:
- `importlib.reload(q06)` only reloads `questions/q06/__init__.py`.
- This package's `run()` reloads its internal run module each call (dev-friendly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from .config import SEED, DATA_ZIP, EXTRACT_DIR, OUTPUT_DIR, PLATFORMS, CUT_LOCAL_TIME
from .extract import extract_dataset
from .pagerank import topk_pagerank_for_platform
from .posts import load_platform_posts, build_activity_features
from .label import classify_top1000
from .report import summary_table

LAST: Dict[str, Any] = {}


def run(output_dir: str = OUTPUT_DIR,
        data_zip: str = DATA_ZIP,
        extract_dir: str = EXTRACT_DIR,
        k: int = 1000) -> Dict[str, Any]:
    """Run the whole Q6 pipeline and save outputs.

    Returns a compact dict with per-platform summaries. Full objects are stored in `LAST`.
    """
    global LAST

    np.random.seed(SEED)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    platform_dirs = extract_dataset(data_zip=data_zip, extract_dir=extract_dir)

    top_pr = {}
    graphs = {}
    activity = {}
    results = {}
    summaries = {}


    for p in PLATFORMS:
        pr_df, G = topk_pagerank_for_platform(platform_dirs[p], k=k, alpha=0.85)
        top_pr[p] = pr_df
        graphs[p] = G

        plat_out = out_root / p
        plat_out.mkdir(parents=True, exist_ok=True)
        pr_df.to_csv(plat_out / f"top{k}_pagerank.csv", index=False)


    for p in PLATFORMS:
        posts_df = load_platform_posts(p, platform_dirs[p], cut_local_time=CUT_LOCAL_TIME)
        feats = build_activity_features(posts_df)
        activity[p] = feats


    for p in PLATFORMS:
        plat_out = out_root / p
        merged = classify_top1000(top_pr[p], activity[p], out_dir=plat_out)
        results[p] = merged
        summ = summary_table(merged).reset_index()
        summaries[p] = summ
        summ.to_csv(plat_out / "summary_by_label.csv", index=False)

    LAST = {
        "output_dir": str(out_root),
        "top_pr": top_pr,
        "graphs": graphs,
        "activity": activity,
        "results": results,
        "summaries": summaries,
    }


    return {
        "output_dir": str(out_root),
        "platforms": PLATFORMS,
        "summaries": {p: summaries[p].to_dict(orient="records") for p in PLATFORMS},
    }
