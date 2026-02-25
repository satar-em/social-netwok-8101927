from __future__ import annotations

import re
import pandas as pd
from .config import PRO_PATTERNS, ANTI_PATTERNS


def stance_from_text(text: str) -> tuple[int, int]:
    pro = sum(1 for pat in PRO_PATTERNS if re.search(pat, text))
    anti = sum(1 for pat in ANTI_PATTERNS if re.search(pat, text))
    return pro, anti


def build_seed_labels(posts_df: pd.DataFrame, min_hits: int = 1, thr: float = 0.4) -> pd.DataFrame:

    agg = posts_df.groupby("user_key")["text"].apply(lambda s: " \n ".join(s.tolist()))
    rows = []
    for user, text in agg.items():
        pro, anti = stance_from_text(text)
        total = pro + anti
        score = 0.0 if total == 0 else (pro - anti) / total
        if total >= min_hits and abs(score) >= thr:
            label = 1 if score > 0 else -1
        else:
            label = 0
        rows.append((user, pro, anti, total, score, label))
    return pd.DataFrame(rows, columns=["user_key", "pro_hits", "anti_hits", "total_hits", "seed_score", "seed_label"])
