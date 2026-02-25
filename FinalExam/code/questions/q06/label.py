"""Rule-based + pseudo-model labeling (ported from Q6 notebook)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd

from .config import SEED, MIN_POSTS_FOR_DECISION


def rule_based_label(feat_row, min_posts17: int = MIN_POSTS_FOR_DECISION) -> Tuple[str, float]:
    if feat_row.get("posts_dey17_total", 0) >= min_posts17:
        if feat_row.get("posts_dey17_after_cut", 0) == 0 and feat_row.get("posts_dey17_before_cut", 0) > 0:
            conf = min(1.0, feat_row.get("posts_dey17_before_cut", 0) / 5.0)
            return "inside_iran", conf
        if feat_row.get("posts_dey17_after_cut", 0) > 0:
            conf = min(1.0, feat_row.get("posts_dey17_after_cut", 0) / 3.0)
            return "outside_iran", conf
    return "unknown", 0.2


def train_pseudo_model(feats: pd.DataFrame):

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except Exception:
        return None, feats

    labels = []
    confs = []
    for _, r in feats.iterrows():
        lab, conf = rule_based_label(r)
        labels.append(lab)
        confs.append(conf)

    feats = feats.copy()
    feats["label_rule"] = labels
    feats["conf_rule"] = confs

    train_df = feats[feats["label_rule"].isin(["inside_iran", "outside_iran"])].copy()
    if train_df.shape[0] < 100:
        return None, feats

    X_cols = [
        "posts_total",
        "posts_pre_15_16",
        "posts_post_18_19",
        "post_pre_ratio",
        "posts_dey17_total",
        "posts_dey17_after_cut",
        "share_dey17_after_cut",
        "hour_mean",
        "hour_std",
    ]
    X = train_df[X_cols].fillna(0).values
    y = train_df["label_rule"].values

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )
    model.fit(X, y)

    X_all = feats[X_cols].fillna(0).values
    proba = model.predict_proba(X_all)
    classes = model.named_steps["clf"].classes_

    preds = []
    confs2 = []
    for i in range(len(feats)):
        p_inside = proba[i, np.where(classes == "inside_iran")[0][0]]
        p_out = proba[i, np.where(classes == "outside_iran")[0][0]]
        if max(p_inside, p_out) >= 0.60:
            if p_inside >= p_out:
                preds.append("inside_iran")
                confs2.append(float(p_inside))
            else:
                preds.append("outside_iran")
                confs2.append(float(p_out))
        else:
            preds.append("unknown")
            confs2.append(float(max(p_inside, p_out)))

    feats["label_model"] = preds
    feats["conf_model"] = confs2
    return model, feats


def classify_top1000(pr_df: pd.DataFrame, feats: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    pr_df = pr_df.copy()
    pr_df["username_norm"] = pr_df["label"].astype(str).str.strip().str.lower()

    model, feats2 = train_pseudo_model(feats)
    merged = pr_df.merge(feats2, on="username_norm", how="left")

    numeric_candidates = [
        "pagerank",
        "views",
        "followers",
        "importance",
        "posts",
        "positive",
        "negative",
        "neutral",
        "posts_total",
        "hour_mean",
        "hour_std",
        "posts_dey17_total",
        "posts_dey17_after_cut",
        "posts_dey17_before_cut",
        "share_dey17_after_cut",
        "posts_pre_15_16",
        "posts_post_18_19",
        "post_pre_ratio",
        "conf_rule",
        "conf_model",
    ]
    numeric_candidates += [c for c in merged.columns if c.startswith("posts_dey_")]

    for c in set(numeric_candidates):
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)

    for c in ["label_rule", "label_model"]:
        if c in merged.columns:
            merged[c] = merged[c].fillna("unknown")

    labs = []
    confs = []
    for _, r in merged.iterrows():
        lab, conf = rule_based_label(r)
        labs.append(lab)
        confs.append(conf)
    merged["label_rule_top"] = labs
    merged["conf_rule_top"] = confs

    if "label_model" in merged.columns:
        merged["final_label"] = merged["label_model"]
        merged["final_conf"] = merged["conf_model"]
    else:
        merged["final_label"] = merged["label_rule_top"]
        merged["final_conf"] = merged["conf_rule_top"]

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "top1000_pagerank_geolocation.csv", index=False)
    return merged
