from ._pairs import _PAIRS_MBART50


def _weights2pairs():
    return {
        'mbart-large-50-many-to-many-mmt': _PAIRS_MBART50,
        'mbart50': _PAIRS_MBART50,
        'facebook/mbart-large-50-many-to-many-mmt': _PAIRS_MBART50
    }

def _dict_from_weights(weights: str) -> dict:
    """Returns a dictionary of lang, codes, pairs if the provided weights is supported."""
    if weights.lower() in _weights2pairs():
        return _weights2pairs[weights.lower()]
    
    else:
        error_message = f"Incorrect argument '{weights}' for parameter weights. Currently, only 'mbart50' is available."
        raise ValueError(error_message)


def get_lang_code_map(weights: str = "mbart50"):
    return _dict_from_weights(weights)["pairs"]


def available_languages(weights: str = "mbart50"):
    return _dict_from_weights(weights)["langs"]


def available_codes(weights: str = "mbart50"):
    return _dict_from_weights(weights)["codes"]
