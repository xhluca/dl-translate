_PAIRS_MBART50 = (
    ("Arabic", "ar_AR"),
    ("Czech", "cs_CZ"),
    ("German", "de_DE"),
    ("English", "en_XX"),
    ("Spanish", "es_XX"),
    ("Estonian", "et_EE"),
    ("Finnish", "fi_FI"),
    ("French", "fr_XX"),
    ("Gujarati", "gu_IN"),
    ("Hindi", "hi_IN"),
    ("Italian", "it_IT"),
    ("Japanese", "ja_XX"),
    ("Kazakh", "kk_KZ"),
    ("Korean", "ko_KR"),
    ("Lithuanian", "lt_LT"),
    ("Latvian", "lv_LV"),
    ("Burmese", "my_MM"),
    ("Nepali", "ne_NP"),
    ("Dutch", "nl_XX"),
    ("Romanian", "ro_RO"),
    ("Russian", "ru_RU"),
    ("Sinhala", "si_LK"),
    ("Turkish", "tr_TR"),
    ("Vietnamese", "vi_VN"),
    ("Chinese", "zh_CN"),
    ("Afrikaans", "af_ZA"),
    ("Azerbaijani", "az_AZ"),
    ("Bengali", "bn_IN"),
    ("Persian", "fa_IR"),
    ("Hebrew", "he_IL"),
    ("Croatian", "hr_HR"),
    ("Indonesian", "id_ID"),
    ("Georgian", "ka_GE"),
    ("Khmer", "km_KH"),
    ("Macedonian", "mk_MK"),
    ("Malayalam", "ml_IN"),
    ("Mongolian", "mn_MN"),
    ("Marathi", "mr_IN"),
    ("Polish", "pl_PL"),
    ("Pashto", "ps_AF"),
    ("Portuguese", "pt_XX"),
    ("Swedish", "sv_SE"),
    ("Swahili", "sw_KE"),
    ("Tamil", "ta_IN"),
    ("Telugu", "te_IN"),
    ("Thai", "th_TH"),
    ("Tagalog", "tl_XX"),
    ("Ukrainian", "uk_UA"),
    ("Urdu", "ur_PK"),
    ("Xhosa", "xh_ZA"),
    ("Galician", "gl_ES"),
    ("Slovene", "sl_SI"),
)

auto_gen_comment = (
    "# Auto-generated. Do not modify, use scripts/generate_lang.py instead.\n"
)

with open("./dl_translate/lang/mbart50.py", "w") as f:
    f.write(auto_gen_comment)
    for lang, code in _PAIRS_MBART50:
        f.write(f'{lang.upper()} = "{lang}"\n')

with open("./dl_translate/_pairs.py", "w") as f:
    f.write(auto_gen_comment)
    f.write(f"_PAIRS_MBART50 = {_PAIRS_MBART50}\n")
