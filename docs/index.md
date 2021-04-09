# User Guide

Quick links:

💻 [GitHub Repository](https://github.com/xhlulu/dl-translate)<br>
📚 [Documentation](https://git.io/dlt-docs) / [Readthedocs](https://dl-translate.readthedocs.io)<br>
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

### Changing the model you are loading

Two model families are available at the moment: [m2m100](https://huggingface.co/transformers/model_doc/m2m_100.html) and [mBART-50 Large](https://huggingface.co/transformers/master/model_doc/mbart.html), which respective allow translation across over 100 languages and 50 languages. By default, the model will select `m2m100`, but you can also explicitly choose the model by specifying the shorthand (`"m2m100"` or `"mbart50"`) or the full repository name (e.g. `"facebook/m2m100_418M"`). For example:

```python
# The following ways are equivalent
mt = dlt.TranslationModel("m2m100")  # Default
mt = dlt.TranslationModel("facebook/m2m100_418M")

# The following ways are equivalent
mt = dlt.TranslationModel("mbart50")
mt = dlt.TranslationModel("facebook/mbart-large-50-many-to-many-mmt")
```

Note that the language code will change depending on the model family. To find out the correct language codes, please read the doc page on available languages or run `mt.available_codes()`.

### Loading from a path

By default, `dlt.TranslationModel` will download the model from the [huggingface repo](https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt) and cache it. If your model is stored locally, you can also directly load that model, but in that case you will need to specify the model family (e.g. `"mbart50"` and `"m2m100"`).

```python
mt = dlt.TranslationModel("/path/to/model/directory/", model_family="mbart50")
# or
mt = dlt.TranslationModel("/path/to/model/directory/", model_family="m2m100")
```
Make sure that your tokenizer is also stored in the same directory if you use this approach.

### Using a different model

You can also choose another model that has the same format as [mbart50](https://huggingface.co/models?filter=mbart-50) or [m2m100](https://huggingface.co/models?search=facebook/m2m100) e.g.
```python
mt = dlt.TranslationModel("facebook/mbart-large-50-one-to-many-mmt", model_family="mbart50")
# or
mt = dlt.TranslationModel("facebook/m2m100_1.2B", model_family="m2m100")
```

Note that the available languages will change if you do this, so you will not be able to leverage `dlt.lang` or `dlt.utils` and the `mt.available_languages()` might also return the incorrect value.

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
import os
import huggingface_hub as hub

dirname = hub.snapshot_download("facebook/m2m100_418M")
os.rename(dirname, "cached_model_m2m100")
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
