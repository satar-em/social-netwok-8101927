from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from .config import Q10Config, ensure_extracted, seed_everything, pandas_defaults
from .io_excel import load_all_posts
from .spread import add_spread_columns, topk_by_platform_day, topk_overall_by_day
from .newsvalues import add_news_values
from .export import export_outputs


def run(
    output_dir: str = "outputs/q10",
    data_zip: str = "data.zip",
    extract_dir: str = "data_extracted",
    platforms: Iterable[str] = ("twitter", "telegram", "instagram"),
    days: Iterable[int] = (15, 16, 17, 18, 19),
    topk: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute Q10 pipeline:
      1) extract data zip (if needed)
      2) load daily Excel files
      3) compute spread proxy and select top-k
      4) tag news values (keyword heuristics)
      5) export CSVs into output_dir
    """
    cfg = Q10Config(
        data_zip=Path(data_zip),
        extract_dir=Path(extract_dir),
        output_dir=Path(output_dir),
        platforms=tuple(platforms),
        days=tuple(days),
        topk=int(topk),
        seed=int(seed),
    )

    seed_everything(cfg.seed)
    pandas_defaults()

    ensure_extracted(cfg.data_zip, cfg.extract_dir)
    data_root = str(cfg.extract_dir)

    all_posts = load_all_posts(data_root, list(cfg.platforms), list(cfg.days))
    all_posts = add_spread_columns(all_posts)

    top_platform_day = topk_by_platform_day(all_posts, k=cfg.topk)
    top_overall_day = topk_overall_by_day(all_posts, k=cfg.topk)

    top_platform_day_nv = add_news_values(top_platform_day)
    top_overall_day_nv = add_news_values(top_overall_day)

    counts = export_outputs(
        out_dir=str(cfg.output_dir),
        top5_overall_by_day_nv=top_overall_day_nv,
        top5_by_platform_day_nv=top_platform_day_nv,
        days=list(cfg.days),
    )

    return {
        "output_dir": str(cfg.output_dir),
        "all_posts_shape": tuple(all_posts.shape),
        "top5_overall_by_day_nv": top_overall_day_nv,
        "top5_by_platform_day_nv": top_platform_day_nv,
        "counts": counts,
    }
