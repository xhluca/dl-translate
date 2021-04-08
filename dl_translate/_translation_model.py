import os
import json
from typing import Union, List, Dict

import transformers
import torch
from tqdm.auto import tqdm

from . import utils


def _select_device(device_selection):
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


def _resolve_lang_codes(source: str, target: str, model_family: str):
    def error_message(variable, value):
        return f'Your {variable}="{value}" is not valid. Please run `print(mt.available_languages())` to see which languages are available.'

    # If can't find in the lang -> code mapping, assumes it's already a code.
    lang_code_map = utils.get_lang_code_map(model_family)
    source = lang_code_map.get(source.capitalize(), source)
    target = lang_code_map.get(target.capitalize(), target)

    # If the code is not valid, raises an error
    if source not in utils.available_codes(model_family):
        raise ValueError(error_message("source", source))
    if target not in utils.available_codes(model_family):
        raise ValueError(error_message("target", target))

    return source, target


def _resolve_tokenizer(model_family):
    di = {
        "mbart50": transformers.MBart50TokenizerFast,
        "m2m100": transformers.M2M100Tokenizer,
    }
    if model_family in di:
        return di[model_family]
    else:
        error_msg = f"{model_family} is not a valid value for model_family. Please choose model_family to be equal to one of the following values: {list(di.keys())}"
        raise ValueError(error_msg)


def _resolve_transformers_model(model_family):
    di = {
        "mbart50": transformers.MBartForConditionalGeneration,
        "m2m100": transformers.M2M100ForConditionalGeneration,
    }
    if model_family in di:
        return di[model_family]
    else:
        error_msg = f"{model_family} is not a valid value for model_family. Please choose model_family to be equal to one of the following values: {list(di.keys())}"
        raise ValueError(error_msg)


def _infer_model_family(model_or_path):
    di = {
        "facebook/mbart-large-50-many-to-many-mmt": "mbart50",
        "facebook/m2m100_418M": "m2m100",
        "facebook/m2m100_1.2B": "m2m100",
    }

    if model_or_path in di:
        return di[model_or_path]
    else:
        error_msg = f'Unable to infer the model_family from "{model_or_path}". Try explicitly setting the value of model_family to "mbart50" or "m2m100".'
        raise ValueError(error_msg)


def _infer_model_or_path(model_or_path):
    di = {
        "mbart50": "facebook/mbart-large-50-many-to-many-mmt",
        "m2m100": "facebook/m2m100_418M",
        "m2m100-small": "facebook/m2m100_418M",
        "m2m100-medium": "facebook/m2m100_1.2B",
    }

    return di.get(model_or_path, model_or_path)


class TranslationModel:
    def __init__(
        self,
        model_or_path: str = "m2m100",
        tokenizer_path: str = None,
        device: str = "auto",
        model_family: str = None,
        model_options: dict = None,
        tokenizer_options: dict = None,
    ):
        """
        *Instantiates a multilingual transformer model for translation.*

        {{params}}
        {{model_or_path}} The path or the name of the model. Equivalent to the first argument of `AutoModel.from_pretrained()`. You can also specify shorthands ("mbart50" and "m2m100").
        {{tokenizer_path}} The path to the tokenizer. By default, it will be set to `model_or_path`.
        {{device}} "cpu", "gpu" or "auto". If it's set to "auto", will try to select a GPU when available or else fall back to CPU.
        {{model_family}} Either "mbart50" or "m2m100". By default, it will be inferred based on `model_or_path`. Needs to be explicitly set if `model_or_path` is a path.
        {{model_options}} The keyword arguments passed to the model, which is a transformer for conditional generation.
        {{tokenizer_options}} The keyword arguments passed to the model's tokenizer.
        """
        model_or_path = _infer_model_or_path(model_or_path)
        self.model_or_path = model_or_path
        self.device = _select_device(device)

        # Resolve default values
        tokenizer_path = tokenizer_path or self.model_or_path
        model_options = model_options or {}
        tokenizer_options = tokenizer_options or {}
        self.model_family = model_family or _infer_model_family(model_or_path)

        # Load the tokenizer
        TokenizerFast = _resolve_tokenizer(self.model_family)
        self._tokenizer = TokenizerFast.from_pretrained(
            tokenizer_path, **tokenizer_options
        )

        # Load the model either from a saved torch model or from transformers.
        if model_or_path.endswith(".pt"):
            self._transformers_model = torch.load(
                model_or_path, map_location=self.device
            ).eval()
        else:
            ModelForConditionalGeneration = _resolve_transformers_model(
                self.model_family
            )
            self._transformers_model = (
                ModelForConditionalGeneration.from_pretrained(
                    self.model_or_path, **model_options
                )
                .to(self.device)
                .eval()
            )

    def translate(
        self,
        text: Union[str, List[str]],
        source: str,
        target: str,
        batch_size: int = 32,
        verbose: bool = False,
        generation_options: dict = None,
    ) -> Union[str, List[str]]:
        """
        *Translates a string or a list of strings from a source to a target language.*

        {{params}}
        {{text}} The content you want to translate.
        {{source}} The language of the original text.
        {{target}} The language of the translated text.
        {{batch_size}} The number of samples to load at once. If set to `None`, it will process everything at once.
        {{verbose}} Whether to display the progress bar for every batch processed.
        {{generation_options}} The keyword arguments passed to `model.generate()`, where `model` is the underlying transformers model.

        Note:
        - Run `print(dlt.utils.available_languages())` to see what's available.
        - A smaller value is preferred for `batch_size` if your (video) RAM is limited.
        """
        if generation_options is None:
            generation_options = {}

        source, target = _resolve_lang_codes(source, target, self.model_family)
        self._tokenizer.src_lang = source

        original_text_type = type(text)
        if original_text_type is str:
            text = [text]

        if batch_size is None:
            batch_size = len(text)

        generation_options.setdefault(
            "forced_bos_token_id", self._tokenizer.lang_code_to_id[target]
        )

        data_loader = torch.utils.data.DataLoader(text, batch_size=batch_size)
        output_text = []

        with torch.no_grad():
            for batch in tqdm(data_loader, disable=not verbose):
                encoded = self._tokenizer(batch, return_tensors="pt", padding=True)
                encoded.to(self.device)

                generated_tokens = self._transformers_model.generate(
                    **encoded, **generation_options
                ).cpu()

                decoded = self._tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )

                output_text.extend(decoded)

        # If text: str and output_text: List[str], then we should convert output_text to str
        if original_text_type is str and len(output_text) == 1:
            output_text = output_text[0]

        return output_text

    def get_transformers_model(self):
        """
        *Retrieve the underlying mBART transformer model.*
        """
        return self._transformers_model

    def get_tokenizer(self):
        """
        *Retrieve the mBART huggingface tokenizer.*
        """
        return self._tokenizer

    def available_languages(self) -> List[str]:
        """
        *Returns all the available languages for a given `dlt.TranslationModel`
        instance.*
        """
        return utils.available_languages(self.model_family)

    def available_codes(self) -> List[str]:
        """
        *Returns all the available codes for a given `dlt.TranslationModel`
        instance.*
        """
        return utils.available_codes(self.model_family)

    def get_lang_code_map(self) -> Dict[str, str]:
        """
        *Returns the language -> codes dictionary for a given `dlt.TranslationModel`
        instance.*
        """
        return utils.get_lang_code_map(self.model_family)

    def save_obj(self, path: str = "saved_model") -> None:
        """
        *Saves your model as a torch object and save your tokenizer.*

        {{params}}
        {{path}} The directory where you want to save your model and tokenizer
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self._transformers_model, os.path.join(path, "weights.pt"))
        self._tokenizer.save_pretrained(path)

        dlt_config = dict(model_family=self.model_family)
        json.dump(dlt_config, open(os.path.join(path, "dlt_config.json"), "w"))

    @classmethod
    def load_obj(cls, path: str = "saved_model", **kwargs):
        """
        *Initialize `dlt.TranslationModel` from the torch object and tokenizer
        saved with `dlt.TranslationModel.save_obj`*

        {{params}}
        {{path}} The directory where your torch model and tokenizer are stored
        """
        config_prev = json.load(open(os.path.join(path, "dlt_config.json"), "rb"))
        config_prev.update(kwargs)
        return cls(
            model_or_path=os.path.join(path, "weights.pt"),
            tokenizer_path=path,
            **config_prev,
        )
