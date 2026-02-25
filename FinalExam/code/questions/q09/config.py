from __future__ import annotations



SEED: int = 42


TOP_N = {
    "twitter": 8000,
    "telegram": 8000,
    "instagram": 8000,
}


PRO_PATTERNS = [
    r"\bلبیک\b",
    r"مدافعان?\s+حرم",
    r"\bسپاه\b",
    r"\bبسیج\b",
    r"خامنه[اآ]ی\s*(عزیز|رهبر)",
    r"حمایت\s+از\s+نظام",
]

ANTI_PATTERNS = [
    r"مرگ\s+بر\s+خامنه[اآ]ی",
    r"مرگ\s+بر\s+دیکتاتور",
    r"زن\s+زندگی\s+آزادی",
    r"جمهوری\s+اسلامی\s+نمی\s*خوایم",
    r"سرنگون(ی|کردن)",
    r"\bپهلوی\b",
]


ANTI_SUBGROUP_HINTS = {
    "monarchist": [r"\bپهلوی\b", r"\bشاه\b", r"\bسلطنت\b", r"\bرضا\s+پهلوی\b"],
    "kurdish":    [r"\bکرد\b", r"\bکورد\b", r"\bکومله\b", r"\bپژاک\b", r"\bروژاوا\b"],
    "baluch":     [r"\bبلوچ\b", r"\bسیستان\b", r"\bزاهدان\b"],
    "azeri":      [r"\bترک\b", r"\bآذربایجان\b", r"\bتبریز\b"],
}


K_ANTI: int = 6
MIN_ANTI_SIZE: int = 80
