from __future__ import annotations

import re

def simple_persian_normalize(text: str) -> str:
    """
    Very light normalization used in the notebook:
    - Arabic ی/ک -> Persian ی/ک
    - keep Persian/word chars, drop punctuation
    - collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
