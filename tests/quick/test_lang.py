import dl_translate as dlt
from dl_translate._pairs import _PAIRS_MBART50


def test_lang():
    for l, _ in _PAIRS_MBART50:
        assert getattr(dlt.lang, l.upper()) == l


def test_lang_mbart50():
    for l, _ in _PAIRS_MBART50:
        assert getattr(dlt.lang.mbart50, l.upper()) == l
