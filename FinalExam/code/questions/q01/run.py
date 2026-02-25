from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .config import METRICS, PLATFORMS, Q1Paths
from .extract import ensure_extracted
from .posts import load_all_posts
from .analysis import (
    export_top_posts,
    export_top_tables,
    topk_accounts_from_gdf,
    build_summaries,
)
from .topics import build_corpus, tfidf_top_terms, lda_topics


def run(
    output_dir: str | Path = "outputs/q01",
    data_zip: str | Path = "data.zip",
    extract_dir: str | Path = "data_unzipped",
    topk: int = 5,
    do_topics: bool = False,
    n_topics: int = 6,
) -> Dict[str, Any]:
    """Run Q1 pipeline and write CSV outputs.

    Returns a dict with in-memory results (top tables, summaries, optional topics).
    """

    here = Path(__file__).resolve()
    project_root = here.parents[2]
    paths = Q1Paths(project_root=project_root)

    out_dir = (project_root / output_dir) if not Path(output_dir).is_absolute() else Path(output_dir)
    data_zip_p = (project_root / data_zip) if not Path(data_zip).is_absolute() else Path(data_zip)
    extract_dir_p = (project_root / extract_dir) if not Path(extract_dir).is_absolute() else Path(extract_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_extracted(data_zip=data_zip_p, extract_dir=extract_dir_p, platforms=PLATFORMS)


    top_tables = {}
    for platform in PLATFORMS:
        tables = topk_accounts_from_gdf(extract_dir_p, platform, metrics=METRICS, topk=topk)
        top_tables[platform] = tables
        export_top_tables(out_dir, platform, tables)


    posts = load_all_posts(extract_dir_p, PLATFORMS)
    summaries = build_summaries(posts, top_tables)

    for platform in PLATFORMS:
        export_top_posts(out_dir, platform, summaries)

    result: Dict[str, Any] = {
        "output_dir": str(out_dir),
        "top_tables": top_tables,
        "summaries": summaries,
    }


    if do_topics:
        topics_out = {}
        for platform in PLATFORMS:
            usernames = sorted({u for s in summaries.get(platform, {}) for u in [s]})
            texts = build_corpus(posts[platform], usernames=usernames)
            topics_out[platform] = {
                "tfidf_top_terms": tfidf_top_terms(texts, topn=20),
                "lda_topics": lda_topics(texts, n_topics=n_topics, n_top_words=10),
            }
        result["topics"] = topics_out

    return result
