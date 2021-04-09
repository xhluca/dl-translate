import dl_translate as dlt
from dl_translate._pairs import _PAIRS_MBART50, _PAIRS_M2M100


def test_lang():
    for l, _ in _PAIRS_M2M100:
        assert getattr(dlt.lang, l.upper().replace(" ", "_")) == l


def test_lang_m2m100():
    for l, _ in _PAIRS_M2M100:
        assert getattr(dlt.lang.m2m100, l.upper().replace(" ", "_")) == l


def test_lang_mbart50():
    for l, _ in _PAIRS_MBART50:
        assert getattr(dlt.lang.mbart50, l.upper().replace(" ", "_")) == l
