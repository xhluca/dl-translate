import torch

import dl_translate as dlt
from dl_translate._translation_model import _resolve_lang_codes, _select_device


def test_resolve_lang_codes_mbart50():
    sources = [dlt.lang.FRENCH, "fr_XX", "French"]
    targets = [dlt.lang.ENGLISH, "en_XX", "English"]

    for source, target in zip(sources, targets):
        s, t = _resolve_lang_codes(source, target, "mbart50")
        assert s == "fr_XX"
        assert t == "en_XX"


def test_resolve_lang_codes_m2m100():
    sources = [dlt.lang.m2m100.FRENCH, "fr", "French"]
    targets = [dlt.lang.m2m100.ENGLISH, "en", "English"]

    for source, target in zip(sources, targets):
        s, t = _resolve_lang_codes(source, target, "m2m100")
        assert s == "fr"
        assert t == "en"


def test_select_device():
    assert _select_device("cpu") == torch.device("cpu")
    assert _select_device("gpu") == torch.device("cuda")
    assert _select_device("cuda:0") == torch.device("cuda", index=0)

    if torch.cuda.is_available():
        assert _select_device("auto") == torch.device("cuda")
    else:
        assert _select_device("auto") == torch.device("cpu")
