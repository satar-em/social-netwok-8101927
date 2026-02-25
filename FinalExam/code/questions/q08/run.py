from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd

from .config import SEED, INCIDENT_DAYS, GDF_PATHS, CLAIMS
from .posts import load_posts_from_zip
from .analysis import (
    build_graph_index,
    match_claims,
    claim_examples,
    claim_accounts,
    claim_communities,
    telegram_multi_claim_communities,
)

def _find_data_zip(candidate: str | Path | None = None) -> Path:
    """Try to locate data.zip with a few sensible defaults."""
    if candidate is not None:
        p = Path(candidate)
        if p.exists():
            return p


    for p in [Path("data.zip"), Path("../data.zip"), Path("../../data.zip"), Path("../../../data.zip")]:
        if p.exists():
            return p

    raise FileNotFoundError(
        "data.zip not found. Provide `data_zip=...` or place data.zip in the project root."
    )

def run(
    out_dir: str | Path = "outputs/q08",
    data_zip: str | Path | None = None,
    incident_days: list[int] | None = None,
) -> dict:
    """Run the Q08 pipeline.

    Outputs (CSV):
      - claim_examples.csv
      - claim_accounts.csv
      - claim_communities.csv
      - telegram_multi_claim_communities.csv
      - factcheck_template.csv

    Returns a small summary dict so you can do: `print(q08.run())`.
    """
    np.random.seed(SEED)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_zip = _find_data_zip(data_zip)


    posts = load_posts_from_zip(data_zip)


    idx = build_graph_index(data_zip=data_zip, gdf_paths=GDF_PATHS, seed=SEED)


    days = incident_days if incident_days is not None else INCIDENT_DAYS
    posts_incident = posts[posts["dey_day"].isin(days)].copy()


    claims_df = match_claims(posts_incident, CLAIMS, idx)


    ex = claim_examples(claims_df)
    ex_path = out_dir / "claim_examples.csv"
    ex.to_csv(ex_path, index=False, encoding="utf-8-sig")

    acct = claim_accounts(claims_df)
    acct_path = out_dir / "claim_accounts.csv"
    acct.to_csv(acct_path, index=False, encoding="utf-8-sig")

    comm = claim_communities(claims_df)
    comm_path = out_dir / "claim_communities.csv"
    comm.to_csv(comm_path, index=False, encoding="utf-8-sig")

    multi = telegram_multi_claim_communities(comm, topn=25)
    multi_path = out_dir / "telegram_multi_claim_communities.csv"
    if not multi.empty:
        multi.to_csv(multi_path, encoding="utf-8-sig")
    else:

        pd.DataFrame().to_csv(multi_path, index=False, encoding="utf-8-sig")

    factcheck = pd.DataFrame(
        [{"claim_key": k, "claim_title": v["title"], "verdict": "", "sources": ""} for k, v in CLAIMS.items()]
    )
    fact_path = out_dir / "factcheck_template.csv"
    factcheck.to_csv(fact_path, index=False, encoding="utf-8-sig")

    return {
        "data_zip": str(data_zip.resolve()),
        "out_dir": str(out_dir.resolve()),
        "posts_shape": tuple(posts.shape),
        "posts_incident_shape": tuple(posts_incident.shape),
        "claims_matched_shape": tuple(claims_df.shape),
        "outputs": {
            "claim_examples": str(ex_path),
            "claim_accounts": str(acct_path),
            "claim_communities": str(comm_path),
            "telegram_multi_claim_communities": str(multi_path),
            "factcheck_template": str(fact_path),
        },
    }
