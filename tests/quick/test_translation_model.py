import dl_translate as dlt
from dl_translate._translation_model import _resolve_lang_codes

def test_resolve_lang_codes():
    sources = [dlt.lang.FRENCH, "fr_XX", "French"]
    targets = [dlt.lang.ENGLISH, "en_XX", "English"]

    for source, target in zip(sources, targets):
        s, t = _resolve_lang_codes(source, target)
        assert s == "fr_XX"
        assert t == "en_XX"