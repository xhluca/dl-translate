from ._pairs import _PAIRS


def get_lang_code_map() -> dict:
    return dict(_PAIRS)

# Variables
available_languages = tuple(pair[0] for pair in _PAIRS)
available_codes = tuple(pair[1] for pair in _PAIRS)
available_langs_and_codes = available_languages + available_codes
