from typing import Union, List

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

from . import utils


def __select_device(device_selection):
    selected = device_selection.lower()
    if selected == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif selected == "cpu":
        device = torch.device("cpu")
    elif selected == "gpu":
        device = torch.device("cuda")
    else:
        device = torch.device(selected)

    return device


def __resolve_lang_codes(source: str, target: str):
    def error_message(variable, value):
        return f'Your {variable}="{value}" is not valid. Please run `print(dlt.utils.available_languages)` to see which languages are available.'

    # If can't find in the lang -> code mapping, assumes it's already a code.
    lang_code_map = utils.get_lang_code_map()
    source = lang_code_map.get(source.capitalize(), source)
    target = lang_code_map.get(target.capitalize(), target)

    # If the code is not valid, raises an error
    if source not in utils.available_codes:
        raise ValueError(error_message("source", source))
    if target not in utils.available_codes:
        raise ValueError(error_message("target", target))

    return source, target


class TranslationModel:
    def __init__(
        self,
        model_or_path: str = "facebook/mbart-large-50-many-to-many-mmt",
        device: str = "auto",
        model_options: dict = {},
        tokenizer_options: dict = {},
    ):
        """Instantiates a multilingual transformer model for translation.
        model_or_path -- The path or the name of the model. Equivalent to the first argument of transformers.AutoModel.from_pretrained().
        device -- "cpu", "gpu" or "auto". If it's set to "auto", will try to select a GPU when available or else fallback to CPU.
        model_options -- The keyword arguments passed to the transformer model, which is a mBART-Large for condition generation.
        tokenizer_options -- The keyword arguments passed to the tokenizer model, which is a mBART-50 Fast Tokenizer.
        """

        model_or_path = model_or_path or "facebook/mbart-large-50-many-to-many-mmt"
        self.device = __select_device(device)

        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            model_or_path, **tokenizer_options
        )
        self.model = MBartForConditionalGeneration.from_pretrained(
            model_or_path, **model_options
        ).to(self.device)

    def translate(
        self,
        text: Union[str, List[str]],
        source: str = "French",
        target: str = "English",
    ) -> Union[str, List[str]]:
        """Translates a string or a list of strings from a source to a target language. Tip: run `print(dlt.utils.available_languages)` to see what's available.
        text -- The content you want to translate.
        source -- The language of the original text.
        target -- The language of the translated text.
        """
        source, target = __resolve_lang_codes(source, target)

        self.tokenizer.src_lang = source

        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)

        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[target],
        ).cpu()
        decoded = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        return decoded

    def get_model(self):
        """Get the mBART transformer model."""
        return self.model

    def get_tokenizer(self):
        """Get the mBART huggingface tokenizer."""
        return self.tokenizer