import dl_translate as dlt
from dl_translate._translation_model import _resolve_lang_codes

s, t = _resolve_lang_codes(dlt.lang.FRENCH, dlt.lang.ENGLISH)
assert s == "French"
assert t == "English"