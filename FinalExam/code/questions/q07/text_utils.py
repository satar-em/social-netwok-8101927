from __future__ import annotations

import re

def normalize_fa(s: str) -> str:
    if not isinstance(s, str):
        return ""

    s = s.replace("ي", "ی").replace("ك", "ک").replace("ة", "ه")

    s = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", s)
    return s

def is_yazd_specific(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(re.search(r"(
