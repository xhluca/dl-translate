# User Guide

Quick links:

💻 [GitHub Repository](https://github.com/xhlulu/dl-translate)<br>
📚 [Documentation](https://xhluca.github.io/dl-translate)<br>
🐍 [PyPi project](https://pypi.org/project/dl-translate/)<br>
🧪 [Colab Demo](https://colab.research.google.com/github/xhlulu/dl-translate/blob/main/demos/colab_demo.ipynb) / [Kaggle Demo](https://www.kaggle.com/xhlulu/dl-translate-demo/)



## Quickstart

Install the library with pip:
```
pip install dl-translate
```

To translate some text:

```python
import dl_translate as dlt

mt = dlt.TranslationModel()  # Slow when you load it for the first time

text_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
mt.translate(text_hi, source=dlt.lang.HINDI, target=dlt.lang.ENGLISH)
```

Above, you can see that `dlt.lang` contains variables representing each of the 50 available languages with auto-complete support. Alternatively, you can specify the language (e.g. "Arabic") or the language code (e.g. "fr" for French):
```python
text_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."
mt.translate(text_ar, source="Arabic", target="fr")
```

If you want to verify whether a language is available, you can check it:
```python
print(mt.available_languages())  # All languages that you can use
print(mt.available_codes())  # Code corresponding to each language accepted
print(mt.get_lang_code_map())  # Dictionary of lang -> code
```

## Usage

### Selecting a device

When you load the model, you can specify the device using the `device` argument. By default, the value will be `device="auto"`, which means it will use a GPU if possible. You can also explicitly set `device="cpu"` or `device="gpu"`, or some other strings accepted by [`torch.device()`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device). __In general, it is recommend to use a GPU if you want a reasonable processing time.__

```python
mt = dlt.TranslationModel(device="auto")  # Automatically select device
mt = dlt.TranslationModel(device="cpu")  # Force you to use a CPU
mt = dlt.TranslationModel(device="gpu")  # Force you to use a GPU
mt = dlt.TranslationModel(device="cuda:2")  # Use the 3rd GPU available
```

### Choosing a different model

By default, the `m2m100` model will be used. However, there are a few options:

* [mBART-50 Large](https://huggingface.co/transformers/master/model_doc/mbart.html):  Allows translations across 50 languages.
* [m2m100](https://huggingface.co/transformers/model_doc/m2m_100.html): Allows translations across 100 languages.
* [nllb-200](https://huggingface.co/docs/transformers/model_doc/nllb) (New in v0.3): Allows translations across 200 languages, and is faster than m2m100 (On RTX A6000, we can see speed up of 3x).

Here's an example:
```python
# The default approval
mt = dlt.TranslationModel("m2m100")  # Shorthand
mt = dlt.TranslationModel("facebook/m2m100_418M")  # Huggingface repo

# If you want to use mBART-50 Large
mt = dlt.TranslationModel("mbart50")
mt = dlt.TranslationModel("facebook/mbart-large-50-many-to-many-mmt")

# Or NLLB-200 (faster and has 200 languages)
mt = dlt.TranslationModel("nllb200")
mt = dlt.TranslationModel("facebook/nllb-200-distilled-600M")
```

Note that the language code will change depending on the model family. To find out the correct language codes, please read the doc page on available languages or run `mt.available_codes()`.

By default, `dlt.TranslationModel` will download the model from the huggingface repo for [mbart50](https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt), [m2m100](https://huggingface.co/facebook/m2m100_418M), or [nllb200](https://huggingface.co/facebook/nllb-200-distilled-600M) and cache it. It's possible to load the model from a path or a model with a similar format, but you will need to specify the `model_family`:
```python
mt = dlt.TranslationModel("/path/to/model/directory/", model_family="mbart50")
mt = dlt.TranslationModel("facebook/m2m100_1.2B", model_family="m2m100")
mt = dlt.TranslationModel("facebook/nllb-200-distilled-600M", model_family="nllb200")
```

Notes:
* Make sure your tokenizer is also stored in the same directory if you load from a file. 
* The available languages will change if you select a different model, so you will not be able to leverage `dlt.lang` or `dlt.utils`.

### Breaking down into sentences

It is not recommended to use extremely long texts as it takes more time to process. Instead, you can try to break them down into sentences. Multiple solutions exists for that, including doing it manually and using the `nltk` library.

A quick approach would be to split them by period. However, you have to ensure that there are no periods used for abbreviations (such as `Mr.` or `Dr.`). For example, it will work in the following case:
```python
text = "Mr Smith went to his favorite cafe. There, he met his friend Dr Doe."
sents = text.split(".")
".".join(mt.translate(sents, source=dlt.lang.ENGLISH, target=dlt.lang.FRENCH))
```


For more complex cases (e.g. where you use periods for abbreviations), you can use `nltk`. First install the library with `pip install nltk`, then run:
```python
import nltk

nltk.download("punkt")

text = "Mr. Smith went to his favorite cafe. There, he met his friend Dr. Doe."
sents = nltk.tokenize.sent_tokenize(text, "english")  # don't use dlt.lang.ENGLISH
" ".join(mt.translate(sents, source=dlt.lang.ENGLISH, target=dlt.lang.FRENCH))
```



### Batch size and verbosity when using `translate`

It's possible to set a batch size (i.e. the number of elements processed at once) for `mt.translate` and whether you want to see the progress bar or not:

```python
...
mt = dlt.TranslationModel()
mt.translate(text, source, target, batch_size=32, verbose=True)
```

If you set `batch_size=None`, it will compute the entire `text` at once rather than splitting into "chunks". We recommend lowering `batch_size` if you do not have a lot of RAM or VRAM and run into CUDA memory error. Set a higher value if you are using a high-end GPU and the VRAM is not fully utilized.


### `dlt.utils` module

An alternative to `mt.available_languages()` is the `dlt.utils` module. You can use it to find out which languages and codes are available:

```python
print(dlt.utils.available_languages('mbart50'))  # All languages that you can use
print(dlt.utils.available_codes('mbart50'))  # Code corresponding to each language accepted
print(dlt.utils.get_lang_code_map('mbart50'))  # Dictionary of lang -> code
print(dlt.utils.available_languages('m2m100'))  # write the name of the model family
```

At the moment, the following models are accepted:
- `"mbart50"`
- `"m2m100"`
- `"nllb200"`

### Offline usage

Unlike the Google translate or MSFT Translator APIs, this library can be fully used offline. However, you will need to first download the packages and models, and move them to your offline environment to be installed and loaded inside a venv.

First, run in your terminal:
```bash
mkdir dlt
cd dlt
mkdir libraries
pip download -d libraries/ dl-translate
```

Once all the required packages are downloaded, you will need to use huggingface hub to download the files. Install it with `pip install huggingface-hub`. Then, run inside Python:
```python
import shutil
import huggingface_hub as hub

dirname = hub.snapshot_download("facebook/m2m100_418M")
shutil.copytree(dirname, "cached_model_m2m100")  # Copy to a permanent folder
```

Now, move everything in the `dlt` directory to your offline environment. Create a virtual environment and run the following in terminal:
```bash
pip install --no-index --find-links libraries/ dl-translate
```

Now, run inside Python:
```python
import dl_translate as dlt

mt = dlt.TranslationModel("cached_model_m2m100", model_family="m2m100")
```

## Advanced

The following section assumes you have knowledge of PyTorch and Huggingface Transformers.

### Saving and loading

If you wish to accelerate the loading time the translation model, you can use `save_obj`. Later you can reload it with `load_obj` by specifying the same directory that you are using to save.

```python
mt = dlt.TranslationModel()
# ...
mt.save_obj('saved_model')
# ...
mt = dlt.TranslationModel.load_obj('saved_model')
```

**Warning:** Only use this if you are certain the torch module saved in `saved_model/weights.pt` can be correctly loaded. Indeed, it is possible that the `huggingface`, `torch` or some other dependencies change between when you called `save_obj` and `load_obj`, and that might break your code. Thus, it is recommend to only run `load_obj` in the same environment/session as `save_obj`. **Note this method might be deprecated in the future once there's no speed benefit in loading this way.**


### Interacting with underlying model and tokenizer

When initializing `model`, you can pass in arguments for the underlying BART model and tokenizer (which will respectively be passed to `ModelForConditionalGeneration.from_pretrained` and `TokenizerFast.from_pretrained`):

```python
mt = dlt.TranslationModel(
    model_options=dict(
        state_dict=...,
        cache_dir=...,
        ...
    ),
    tokenizer_options=dict(
        tokenizer_file=...,
        eos_token=...,
        ...
    )
)
```

You can also access the underlying `transformers` model and `tokenizer`:
```python
transformers_model = mt.get_transformers_model()
tokenizer = mt.get_tokenizer()
```

For more information about the models themselves, please read the docs on [mBART](https://huggingface.co/transformers/master/model_doc/mbart.html) and [m2m100](https://huggingface.co/transformers/model_doc/m2m_100.html).


### Keyword arguments for the `generate()` method of the underlying model

When running `mt.translate`, you can also give a `generation_options` dictionary that is passed as keyword arguments to the underlying `mt.get_transformers_model().generate()` method:
```python
mt.translate(
    text,
    source=dlt.lang.GERMAN,
    target=dlt.lang.SPANISH,
    generation_options=dict(num_beams=5, max_length=...)
)
```

Learn more in the [huggingface docs](https://huggingface.co/transformers/main_classes/model.html#transformers.generation_utils.GenerationMixin.generate).
