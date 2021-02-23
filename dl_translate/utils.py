from ._pairs import _PAIRS


__LANGS = tuple(pair[0] for pair in _PAIRS)
__CODES = tuple(pair[1] for pair in _PAIRS)


def get_lang_code_map() -> dict:
    return dict(_PAIRS)


def available_languages():
    return __LANGS


def available_codes():
    return __CODES