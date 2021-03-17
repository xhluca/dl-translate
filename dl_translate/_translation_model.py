import os
from typing import Union, List, Dict

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
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


def _resolve_lang_codes(source: str, target: str):
    def error_message(variable, value):
        return f'Your {variable}="{value}" is not valid. Please run `print(dlt.utils.available_languages())` to see which languages are available.'

    # If can't find in the lang -> code mapping, assumes it's already a code.
    lang_code_map = utils.get_lang_code_map()
    source = lang_code_map.get(source.capitalize(), source)
    target = lang_code_map.get(target.capitalize(), target)

    # If the code is not valid, raises an error
    if source not in utils.available_codes():
        raise ValueError(error_message("source", source))
    if target not in utils.available_codes():
        raise ValueError(error_message("target", target))

    return source, target


class TranslationModel:
    def __init__(
        self,
        model_or_path: str = "facebook/mbart-large-50-many-to-many-mmt",
        tokenizer_path: str = None,
        device: str = "auto",
        model_options: dict = None,
        tokenizer_options: dict = None,
    ):
        """
        Instantiates a multilingual transformer model for translation.

        {{params}}
        {{model_or_path}} The path or the name of the model. Equivalent to the first argument of AutoModel.from_pretrained().
        {{device}} "cpu", "gpu" or "auto". If it's set to "auto", will try to select a GPU when available or else fallback to CPU.
        {{tokenizer_path}} The path to the tokenizer, only if it is different from `model_or_path`; otherwise, leave it as `None`.
        {{model_options}} The keyword arguments passed to the transformer model, which is a mBART-Large for condition generation.
        {{tokenizer_options}} The keyword arguments passed to the tokenizer model, which is a mBART-50 Fast Tokenizer.
        """
        self.model_or_path = model_or_path
        self.device = _select_device(device)

        # Resolve default values
        tokenizer_path = tokenizer_path or self.model_or_path
        model_options = model_options or {}
        tokenizer_options = tokenizer_options or {}

        self.tokenizer = MBart50TokenizerFast.from_pretrained(
            tokenizer_path, **tokenizer_options
        )

        if model_or_path.endswith(".pt"):
            self.bart_model = torch.load(model_or_path, map_location=self.device).eval()
        else:
            self.bart_model = (
                MBartForConditionalGeneration.from_pretrained(
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
        {{batch_size}} The number of samples to load at once. A smaller value is preferred if you do not have a lot of (video) RAM. If set to `None`, it will process everything at once.
        {{verbose}} Whether to display the progress bar for every batch processed.
        {{generation_options}} The keyword arguments passed to bart_model.generate(), where bart_model is the underlying transformers model.

        Tip: run `print(dlt.utils.available_languages())` to see what's available.
        """
        if generation_options is None:
            generation_options = {}

        source, target = _resolve_lang_codes(source, target)
        self.tokenizer.src_lang = source

        original_text_type = type(text)
        if original_text_type is str:
            text = [text]

        if batch_size is None:
            batch_size = len(text)

        generation_options.setdefault(
            "forced_bos_token_id", self.tokenizer.lang_code_to_id[target]
        )

        data_loader = torch.utils.data.DataLoader(text, batch_size=batch_size)
        output_text = []

        with torch.no_grad():
            for batch in tqdm(data_loader, disable=not verbose):
                encoded = self.tokenizer(batch, return_tensors="pt", padding=True)
                encoded.to(self.device)

                generated_tokens = self.bart_model.generate(
                    **encoded, **generation_options
                ).cpu()

                decoded = self.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )

                output_text.extend(decoded)

        # If text: str and output_text: List[str], then we should convert output_text to str
        if original_text_type is str and len(output_text) == 1:
            output_text = output_text[0]

        return output_text

    def get_transformers_model(self) -> MBartForConditionalGeneration:
        """
        *Retrieve the underlying mBART transformer model.*
        """
        return self.bart_model

    def get_tokenizer(self) -> MBart50TokenizerFast:
        """
        *Retrieve the mBART huggingface tokenizer.*
        """
        return self.tokenizer

    def available_languages(self) -> List[str]:
        """
        *Returns all the available languages for a given `dlt.TranslationModel`
        instance.*
        """
        return utils.available_languages("mbart50")

    def available_codes(self) -> List[str]:
        """
        *Returns all the available codes for a given `dlt.TranslationModel`
        instance.*
        """
        return utils.available_languages("mbart50")

    def get_lang_code_map(self) -> Dict[str, str]:
        """
        *Returns the language -> codes dictionary for a given `dlt.TranslationModel`
        instance.*
        """
        return utils.get_lang_code_map("mbart50")

    def save_obj(self, path: str = "saved_model") -> None:
        """
        *Saves your model as a torch object and save your tokenizer.*

        {{params}}
        {{path}} The directory where you want to save your model and tokenizer
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.bart_model, os.path.join(path, "weights.pt"))
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load_obj(cls, path: str = "saved_model", **kwargs):
        """
        *Initialize `dlt.TranslationModel` from the torch object and tokenizer
        saved with `dlt.TranslationModel.save_obj`*

        {{params}}
        {{path}} The directory where your torch model and tokenizer are stored
        """
        load_dir = os.path.join(path, "weights.pt")
        return cls(model_or_path=load_dir, tokenizer_path=path, **kwargs)
