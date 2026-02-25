from __future__ import annotations

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import ANTI_SUBGROUP_HINTS


def top_terms_for_faction(platform: str, faction_id: int, graphs: dict, factions: dict, posts: dict, topn: int = 25):
    Gd, _ = graphs[platform]
    node2label = {n: str(d.get("label", "")).strip().lower().lstrip("@") for n, d in Gd.nodes(data=True)}

    nodes_in = [n for n, fid in factions[platform].items() if fid == faction_id]
    labels_in = set(node2label[n] for n in nodes_in if node2label[n])

    df = posts[platform]
    sub = df[df["user_key"].isin(labels_in)].copy()
    if sub.empty:
        return {"n_users_with_text": 0, "top_terms": [], "hint_counts": {}}

    user_text = sub.groupby("user_key")["text"].apply(lambda s: " \n ".join(s.tolist()))
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(user_text.values)
    tfidf_mean = np.asarray(X.mean(axis=0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    top_idx = np.argsort(-tfidf_mean)[:topn]
    top_terms = list(zip(vocab[top_idx].tolist(), tfidf_mean[top_idx].tolist()))

    hint_counts = {}
    joined = " \n ".join(user_text.values)
    for name, pats in ANTI_SUBGROUP_HINTS.items():
        hint_counts[name] = sum(1 for pat in pats if re.search(pat, joined))

    return {
        "n_users_with_text": int(user_text.shape[0]),
        "top_terms": top_terms,
        "hint_counts": hint_counts,
    }
