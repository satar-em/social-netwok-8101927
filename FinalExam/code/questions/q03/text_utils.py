from __future__ import annotations

import re


PERSIAN_STOPWORDS = set('''
و در به از که را با برای این آن یک یا اما اگر نه من تو او ما شما آنها
ها های هم نیز ولی فقط خیلی بیشتر کمتر چون چرا وقتی سپس همچنین
می کنم کنی کند کنیم کنید کنند شده شد شود بوده بود باشند هست نیست
'''.split())

def normalize_persian_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()

    s = re.sub(r"http\S+|www\.\S+", " ", s)

    s = re.sub(r"[^0-9a-z\u0600-\u06FF\s]+", " ", s)

    s = re.sub(r"\s+", " ", s).strip()
    return s

