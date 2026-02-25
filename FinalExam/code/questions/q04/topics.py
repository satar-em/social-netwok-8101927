from __future__ import annotations

import numpy as np

from .config import SEED
from .posts import load_all_excels

def community_keywords_ml(platform: str, comm_nodes, nodes_df,
                          extract_dir,
                          top_terms: int = 15, n_topics: int = 3, min_docs: int = 50):
    """
    Bonus (Notebook): community keywords via TF-IDF + NMF.
    Returns dict or None if not enough docs.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import NMF
    except Exception as e:
        raise ImportError("scikit-learn is required for community_keywords_ml") from e

    posts = load_all_excels(extract_dir=extract_dir, platform=platform)

    id2user = nodes_df.set_index("name")["label"].to_dict()
    usernames = [id2user.get(n) for n in comm_nodes]
    usernames = [u for u in usernames if isinstance(u, str) and len(u) > 0]

    dfc = posts[posts["username"].isin(usernames)].copy()
    texts = dfc["text"].dropna().tolist()

    if len(texts) < min_docs:
        return None

    vectorizer = TfidfVectorizer(
        max_features=20000,
        min_df=3,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(texts)

    nmf = NMF(n_components=n_topics, random_state=SEED)
    nmf.fit(X)
    H = nmf.components_
    vocab = np.array(vectorizer.get_feature_names_out())

    topics = []
    for t in range(n_topics):
        top_idx = np.argsort(H[t])[::-1][:top_terms]
        terms = vocab[top_idx].tolist()
        topics.append(terms)

    return {"platform": platform, "docs": len(texts), "topics": topics}
