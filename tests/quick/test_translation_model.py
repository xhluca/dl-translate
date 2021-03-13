import torch

import dl_translate as dlt
from dl_translate._translation_model import _resolve_lang_codes, _select_device


def test_resolve_lang_codes():
    sources = [dlt.lang.FRENCH, "fr_XX", "French"]
    targets = [dlt.lang.ENGLISH, "en_XX", "English"]

    for source, target in zip(sources, targets):
        s, t = _resolve_lang_codes(source, target)
        assert s == "fr_XX"
        assert t == "en_XX"


def test_select_device():
    assert _select_device("cpu") == torch.device("cpu")
    assert _select_device("gpu") == torch.device("cuda")
    assert _select_device("cuda:0") == torch.device("cuda", index=0)

    if torch.cuda.is_available():
        assert _select_device("auto") == torch.device("cuda")
    else:
        assert _select_device("auto") == torch.device("cpu")
