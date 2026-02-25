from __future__ import annotations


SEED = 42


INCIDENT_DAYS = list(range(15, 20))


GDF_PATHS = {
    "twitter":   "twitter/twitter-10 to 24 dey.gdf",
    "telegram":  "telegram/telegram-10 to 24 dey.gdf",
    "instagram": "instagram/instagram-10 to 24 dey.gdf",
}


CLAIMS = {
    "khamenei_escape_mehrabad": {
        "title": "«آماده‌باش اضطراری مهرآباد برای فرار خامنه‌ای به مسکو»",
        "pattern": r"(مهراباد|مهرآباد).*(پرواز|اضطراری|فرار).*(خامنه|مسکو)|فرار.*خامنه.*مسکو|جت اختصاصی.*رهبر|Antonov|An-124",
    },
    "police_joined_protesters": {
        "title": "«پیوستن نیروی انتظامی به معترضان»",
        "pattern": r"پیوستن.*نیروی انتظامی|نیروی انتظامی.*پیوست|پلیس.*به معترضان پیوست",
    },
    "us_tanker_over_iran": {
        "title": "«ورود/پرواز هواپیمای سوخت‌رسان آمریکا به داخل خاک ایران (مثلاً بر فراز سنندج)»",
        "pattern": r"(سوخت[\s‌-]*رسان|KC-135|KC135).*(ایران|خاک ایران|سنندج|کردستان)|سنندج.*سوخت[\s‌-]*رسان",
    },
    "turkey_evacuating": {
        "title": "«ترکیه در حال خارج‌کردن/تخلیه اتباع خود از ایران است»",
        "pattern": r"اتباع ترکیه|تخلیه.*ترکیه|خارج.*اتباع ترکیه|اتباع.*ترکیه.*خارج",
    },
    "sarpol_two_killed": {
        "title": "«کشته‌شدن 2 نفر در سرپل‌ذهاب در ناآرامی‌ها»",
        "pattern": r"تکذیب.*کشته شدن.*(2|۲|دو).*نفر.*سرپل|سرپل[\s‌-]*ذهاب.*(کشته شدن|کشته‌شدن).*(2|۲|دو).*(نفر)",
    },
}
