from __future__ import annotations

import os
import pandas as pd


BASE_COLS = [
    "day_dey","platform","username","name","date","date_jalali","link",
    "spread_score","forward","engagement","impression","like","comment","sentiment","text",
]

NV_COLS = [
    "timeliness","negativity","conflict","impact_magnitude","elite_prominence",
    "proximity","surprise","human_interest","visuals","shareability",
]


def export_outputs(
    out_dir: str,
    top5_overall_by_day_nv: pd.DataFrame,
    top5_by_platform_day_nv: pd.DataFrame,
    days: list[int],
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    top5_overall_by_day_nv[BASE_COLS + NV_COLS].to_csv(
        os.path.join(out_dir, "top5_overall_by_day_with_newsvalues.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    top5_by_platform_day_nv[BASE_COLS + NV_COLS].to_csv(
        os.path.join(out_dir, "top5_by_platform_day_with_newsvalues.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    counts = top5_overall_by_day_nv.groupby("day_dey")[NV_COLS].sum().astype(int)
    counts.to_csv(
        os.path.join(out_dir, "newsvalues_counts_overall_top5_by_day.csv"),
        encoding="utf-8-sig",
    )

    for d in days:
        df_day = top5_overall_by_day_nv[top5_overall_by_day_nv["day_dey"] == d].copy()
        df_day["text_excerpt"] = (
            df_day["text"].astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.slice(0, 180)
        )
        cols = ["platform", "username", "spread_score", "text_excerpt"] + NV_COLS
        df_day[cols].to_csv(
            os.path.join(out_dir, f"day_{d}_overall_top5_compact.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    return counts
