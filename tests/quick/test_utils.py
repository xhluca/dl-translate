import pytest

from dl_translate import utils
from dl_translate._pairs import _PAIRS_MBART50


def test_dict_from_weights():
    weights = ["mbart50", "mbart-large-50-many-to-many-mmt"]

    valid_keys = ["langs", "codes", "pairs"]

    for w in weights:
        assert type(utils._dict_from_weights(w)) is dict

        keys = utils._dict_from_weights(w).keys()
        for key in valid_keys:
            assert key in keys


def test_dict_from_weights_exception():
    weights = ["mbart50", "mbart-large-50-many-to-many-mmt"]

    valid_keys = ["langs", "codes", "pairs"]

    with pytest.raises(ValueError):
        utils._dict_from_weights("incorrect")


def test_available_languages():
    langs = utils.available_languages()

    for lang, _ in _PAIRS_MBART50:
        assert lang in langs


def test_available_codes():
    codes = utils.available_codes()

    for _, code in _PAIRS_MBART50:
        assert code in codes
