from typing import Dict, List

from ._pairs import _PAIRS_MBART50, _PAIRS_M2M100


def _weights2pairs():
    return {
        "mbart50": _PAIRS_MBART50,
        "mbart-large-50-many-to-many-mmt": _PAIRS_MBART50,
        "facebook/mbart-large-50-many-to-many-mmt": _PAIRS_MBART50,
        "m2m100": _PAIRS_M2M100,
        "m2m100_418M": _PAIRS_M2M100,
        "m2m100_1.2B": _PAIRS_M2M100,
        "facebook/m2m100_418M": _PAIRS_M2M100,
        "facebook/m2m100_1.2B": _PAIRS_M2M100,
    }


def _dict_from_weights(weights: str) -> dict:
    """Returns a dictionary of lang, codes, pairs if the provided weights is supported."""
    if weights in _weights2pairs():
        pairs = _weights2pairs()[weights]
        return {
            "langs": tuple(pair[0] for pair in pairs),
            "codes": tuple(pair[1] for pair in pairs),
            "pairs": dict(pairs),
        }
    elif weights.lower() in _weights2pairs():
        pairs = _weights2pairs()[weights.lower()]
        return {
            "langs": tuple(pair[0] for pair in pairs),
            "codes": tuple(pair[1] for pair in pairs),
            "pairs": dict(pairs),
        }

    else:
        error_message = f"Incorrect argument '{weights}' for parameter weights. Please choose from: {list(_weights2pairs().keys())}"
        raise ValueError(error_message)


def get_lang_code_map(weights: str = "mbart50") -> Dict[str, str]:
    """
    *Get a dictionary mapping a language -> code for a given model. The code will depend on the model you choose.*

    {{params}}
    {{weights}} The name of the model you are using. For example, "mbart50" is the multilingual BART Large with 50 languages available to use.
    """
    return _dict_from_weights(weights)["pairs"]


def available_languages(weights: str = "mbart50") -> List[str]:
    """
    *Get all the languages available for a given model.*

    {{params}}
    {{weights}} The name of the model you are using. For example, "mbart50" is the multilingual BART Large with 50 languages available to use.
    """
    return _dict_from_weights(weights)["langs"]


def available_codes(weights: str = "mbart50") -> List[str]:
    """
    *Get all the codes available for a given model. The code format will depend on the model you select.*

    {{params}}
    {{weights}} The name of the model you are using. For example, "mbart50" is the multilingual BART Large with 50 codes available to use.
    """
    return _dict_from_weights(weights)["codes"]
