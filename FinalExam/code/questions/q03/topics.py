from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

from .text_utils import normalize_persian_text, PERSIAN_STOPWORDS

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception as e:
    TfidfVectorizer = None

def tfidf_keywords_by_user(df: pd.DataFrame, users: List[str], top_k: int = 15) -> pd.DataFrame:
    'Build one document per user and extract top TF-IDF keywords.'
    user_docs = []
    user_list = []

    for u in users:
        sub = df[df["username"] == u]
        texts = [normalize_persian_text(t) for t in sub.get("text", [])]
        texts = [t for t in texts if t]
        if not texts:
            continue
        user_list.append(u)
        user_docs.append(" ".join(texts))

    if not user_docs:
        return pd.DataFrame(columns=["username", "top_keywords", "num_posts_in_excels"])

    vectorizer = TfidfVectorizer(
        stop_words=list(PERSIAN_STOPWORDS),
        max_features=5000,
        ngram_range=(1,2),
        min_df=2
    )
    X = vectorizer.fit_transform(user_docs)
    vocab = np.array(vectorizer.get_feature_names_out())

    rows = []
    for i, u in enumerate(user_list):
        scores = X[i].toarray().ravel()
        top_idx = np.argsort(scores)[::-1][:top_k]
        kw = [vocab[j] for j in top_idx if scores[j] > 0]
        rows.append({
            "username": u,
            "top_keywords": ", ".join(kw),
            "num_posts_in_excels": int((df["username"] == u).sum())
        })

    return pd.DataFrame(rows).sort_values(["num_posts_in_excels","username"], ascending=[False, True]).reset_index(drop=True)

def _require_sklearn():
    if TfidfVectorizer is None:
        raise ImportError("scikit-learn is required for TF-IDF topic analysis. Install: pip install scikit-learn")
