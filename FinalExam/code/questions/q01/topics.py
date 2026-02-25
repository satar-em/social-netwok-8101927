from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

from .text_utils import normalize_text


def build_corpus(df: pd.DataFrame, usernames: Optional[Sequence[str]] = None, min_chars: int = 3) -> List[str]:
    if usernames is not None:
        sub = df[df["username"].astype(str).str.lower().isin([u.lower() for u in usernames])].copy()
    else:
        sub = df.copy()
    texts = [normalize_text(t) for t in sub["text"].astype(str).tolist()] if "text" in sub.columns else []
    texts = [t for t in texts if len(t) >= min_chars]
    return texts


def tfidf_top_terms(texts: List[str], topn: int = 20, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X = vec.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    top_idx = np.argsort(scores)[::-1][:topn]
    return list(zip(terms[top_idx].tolist(), scores[top_idx].tolist()))


def lda_topics(texts: List[str], n_topics: int = 8, n_top_words: int = 12, max_features: int = 6000):
    vec = TfidfVectorizer(max_features=max_features)
    X = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0, learning_method="batch")
    lda.fit(X)
    terms = np.array(vec.get_feature_names_out())
    topics = []
    for k, comp in enumerate(lda.components_):
        top_idx = np.argsort(comp)[::-1][:n_top_words]
        topics.append((int(k), terms[top_idx].tolist()))
    return topics
