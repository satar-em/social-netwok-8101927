from __future__ import annotations

import collections
import re
from typing import Iterable, List, Tuple

url_re = re.compile(r"https?://\S+|www\.\S+")

PERSIAN_STOPWORDS = set(
    '''و در به از که را با برای این آن یک یا تا اما اگر چون نیز هم همه ما شما او آنها من تو
است هست بودند بود شدن شده شود می کنم کنیم کنید کرده کردن دارد دارند
ها های ی یی ای نه'''.split()
)
PERSIAN_STOPWORDS.update(["می","کرد","کرده","شود","شده","است","هست","بود","های","ها","ای"])


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = url_re.sub("", s)
    s = s.replace("\u200c", " ")
    s = re.sub(r"[^\u0600-\u06FF0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def top_tokens(texts: Iterable[str], n: int = 15) -> List[Tuple[str, int]]:
    cnt = collections.Counter()
    for t in texts:
        t = normalize_text(t)
        for tok in t.split():
            if len(tok) < 2:
                continue
            if tok in PERSIAN_STOPWORDS:
                continue
            cnt[tok] += 1
    return cnt.most_common(n)
