# API Reference


## dlt.TranslationModel


### __init__

```python
dlt.TranslationModel.__init__(self, model_or_path: str = 'm2m100', tokenizer_path: str = None, device: str = 'auto', model_family: str = None, model_options: dict = None, tokenizer_options: dict = None)
```

*Instantiates a multilingual transformer model for translation.*

| Parameter | Type | Default | Description |
|-|-|-|-|
| **model_or_path** | *str* | `m2m100` | The path or the name of the model. Equivalent to the first argument of `AutoModel.from_pretrained()`. You can also specify shorthands ("mbart50" and "m2m100").
| **tokenizer_path** | *str* | *optional* | The path to the tokenizer. By default, it will be set to `model_or_path`.
| **device** | *str* | `auto` | "cpu", "gpu" or "auto". If it's set to "auto", will try to select a GPU when available or else fall back to CPU.
| **model_family** | *str* | *optional* | Either "mbart50" or "m2m100". By default, it will be inferred based on `model_or_path`. Needs to be explicitly set if `model_or_path` is a path.
| **model_options** | *dict* | *optional* | The keyword arguments passed to the model, which is a transformer for conditional generation.
| **tokenizer_options** | *dict* | *optional* | The keyword arguments passed to the model's tokenizer.

<br>


### translate

```python
dlt.TranslationModel.translate(self, text: Union[str, List[str]], source: str, target: str, batch_size: int = 32, verbose: bool = False, generation_options: dict = None) -> Union[str, List[str]]
```

*Translates a string or a list of strings from a source to a target language.*

| Parameter | Type | Default | Description |
|-|-|-|-|
| **text** | *Union[str, List[str]]* | *required* | The content you want to translate.
| **source** | *str* | *required* | The language of the original text.
| **target** | *str* | *required* | The language of the translated text.
| **batch_size** | *int* | `32` | The number of samples to load at once. If set to `None`, it will process everything at once.
| **verbose** | *bool* | `False` | Whether to display the progress bar for every batch processed.
| **generation_options** | *dict* | *optional* | The keyword arguments passed to `model.generate()`, where `model` is the underlying transformers model.

Note:
- Run `print(dlt.utils.available_languages())` to see what's available.
- A smaller value is preferred for `batch_size` if your (video) RAM is limited.

<br>


### get_transformers_model

```python
dlt.TranslationModel.get_transformers_model(self)
```

*Retrieve the underlying mBART transformer model.*

<br>


### get_tokenizer

```python
dlt.TranslationModel.get_tokenizer(self)
```

*Retrieve the mBART huggingface tokenizer.*

<br>


### available_codes

```python
dlt.TranslationModel.available_codes(self) -> List[str]
```

*Returns all the available codes for a given `dlt.TranslationModel`
instance.*

<br>


### available_languages

```python
dlt.TranslationModel.available_languages(self) -> List[str]
```

*Returns all the available languages for a given `dlt.TranslationModel`
instance.*

<br>


### get_lang_code_map

```python
dlt.TranslationModel.get_lang_code_map(self) -> Dict[str, str]
```

*Returns the language -> codes dictionary for a given `dlt.TranslationModel`
instance.*

<br>


### save_obj

```python
dlt.TranslationModel.save_obj(self, path: str = 'saved_model') -> None
```

*Saves your model as a torch object and save your tokenizer.*

| Parameter | Type | Default | Description |
|-|-|-|-|
| **path** | *str* | `saved_model` | The directory where you want to save your model and tokenizer

<br>


### load_obj

```python
dlt.TranslationModel.load_obj(path: str = 'saved_model', **kwargs)
```

*Initialize `dlt.TranslationModel` from the torch object and tokenizer
saved with `dlt.TranslationModel.save_obj`*

| Parameter | Type | Default | Description |
|-|-|-|-|
| **path** | *str* | `saved_model` | The directory where your torch model and tokenizer are stored

<br>



<br>


## dlt.utils


### get_lang_code_map

```python
dlt.utils.get_lang_code_map(weights: str = 'mbart50') -> Dict[str, str]
```

*Get a dictionary mapping a language -> code for a given model. The code will depend on the model you choose.*

| Parameter | Type | Default | Description |
|-|-|-|-|
| **weights** | *str* | `mbart50` | The name of the model you are using. For example, "mbart50" is the multilingual BART Large with 50 languages available to use.

<br>


### available_codes

```python
dlt.utils.available_codes(weights: str = 'mbart50') -> List[str]
```

*Get all the codes available for a given model. The code format will depend on the model you select.*

| Parameter | Type | Default | Description |
|-|-|-|-|
| **weights** | *str* | `mbart50` | The name of the model you are using. For example, "mbart50" is the multilingual BART Large with 50 codes available to use.

<br>


### available_languages

```python
dlt.utils.available_languages(weights: str = 'mbart50') -> List[str]
```

*Get all the languages available for a given model.*

| Parameter | Type | Default | Description |
|-|-|-|-|
| **weights** | *str* | `mbart50` | The name of the model you are using. For example, "mbart50" is the multilingual BART Large with 50 languages available to use.

<br>



<br>

