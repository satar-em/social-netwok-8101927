from __future__ import annotations

import re

def normalize_fa(s: str) -> str:
    """Basic Persian/Arabic normalization used in the notebook."""
    if not isinstance(s, str):
        return ""
    s = s.replace("ي", "ی").replace("ك", "ک").replace("ة", "ه")

    s = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_handle(u: str) -> str:
    """Normalize usernames/handles for matching against graph node labels."""
    if not isinstance(u, str):
        return ""
    u = u.strip()
    if u.startswith("@"):
        u = u[1:]
    return u.lower()
