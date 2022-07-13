# DL Translate

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5230676.svg)](https://doi.org/10.5281/zenodo.5230676)
[![Downloads](https://static.pepy.tech/personalized-badge/dl-translate?period=total&units=abbreviation&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/dl-translate)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/xhluca/dl-translate/blob/main/LICENSE)


*A deep learning-based translation library built on Huggingface `transformers`*

💻 [GitHub Repository](https://github.com/xhluca/dl-translate)<br>
📚 [Documentation](https://xhluca.github.io/dl-translate) / [Readthedocs](https://dl-translate.readthedocs.io)<br>
🐍 [PyPi project](https://pypi.org/project/dl-translate/)<br>
🧪 [Colab Demo](https://colab.research.google.com/github/xhluca/dl-translate/blob/main/demos/colab_demo.ipynb) / [Kaggle Demo](https://www.kaggle.com/xhlulu/dl-translate-demo/)



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

When you load the model, you can specify the device:
```python
mt = dlt.TranslationModel(device="auto")
```

By default, the value will be `device="auto"`, which means it will use a GPU if possible. You can also explicitly set `device="cpu"` or `device="gpu"`, or some other strings accepted by [`torch.device()`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device). __In general, it is recommend to use a GPU if you want a reasonable processing time.__

### Choosing a different model

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

By default, `dlt.TranslationModel` will download the model from the huggingface repo for [mbart50](https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt) or [m2m100](https://huggingface.co/facebook/m2m100_418M) and cache it. It's possible to load the model from a path or a model with a similar format, but you will need to specify the `model_family`:
```python
mt = dlt.TranslationModel("/path/to/model/directory/", model_family="mbart50")
mt = dlt.TranslationModel("facebook/m2m100_1.2B", model_family="m2m100")
```

Notes:
* Make sure your tokenizer is also stored in the same directory if you load from a file. 
* The available languages will change if you select a different model, so you will not be able to leverage `dlt.lang` or `dlt.utils`.

### Splitting into sentences

It is not recommended to use extremely long texts as it takes more time to process. Instead, you can try to break them down into sentences with the help of `nltk`. First install the library with `pip install nltk`, then run:
```python
import nltk

nltk.download("punkt")

text = "Mr. Smith went to his favorite cafe. There, he met his friend Dr. Doe."
sents = nltk.tokenize.sent_tokenize(text, "english")  # don't use dlt.lang.ENGLISH
" ".join(mt.translate(sents, source=dlt.lang.ENGLISH, target=dlt.lang.FRENCH))
```

### Batch size during translation

It's possible to set a batch size (i.e. the number of elements processed at once) for `mt.translate` and whether you want to see the progress bar or not:

```python
# ...
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
```

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

If you have knowledge of PyTorch and Huggingface Transformers, you can access advanced aspects of the library for more customization:
* **Saving and loading**: If you wish to accelerate the loading time the translation model, you can use `save_obj` and reload it later with `load_obj`. This method is only recommended if you are familiar with `huggingface` and `torch`; please read the docs for more information.
* **Interacting with underlying model and tokenizer**: When initializing `model`, you can pass in arguments for the underlying BART model and tokenizer with `model_options` and `tokenizer_options` respectively. You can also access the underlying `transformers` with `mt.get_transformers_model()`.
* **Keyword arguments for the `generate()` method**: When running `mt.translate`, you can also give `generation_options` that is passed to the `generate()` method of the underlying transformer model.

For more information, please visit the [advanced section of the user guide](https://xhluca.github.io/dl-translate/#advanced) (also available in the [readthedocs version](https://dl-translate.readthedocs.io/en/latest/#advanced)).

## Acknowledgement

`dl-translate` is built on top of Huggingface's implementation of two models created by Facebook AI Research.

1. The multilingual BART finetuned on many-to-many translation of over 50 languages, which is [documented here](https://huggingface.co/transformers/master/model_doc/mbart.html) The original paper was written by Tang et. al from Facebook AI Research; you can [find it here](https://arxiv.org/pdf/2008.00401.pdf) and cite it using the following:
    ```
    @article{tang2020multilingual,
        title={Multilingual translation with extensible multilingual pretraining and finetuning},
        author={Tang, Yuqing and Tran, Chau and Li, Xian and Chen, Peng-Jen and Goyal, Naman and Chaudhary, Vishrav and Gu, Jiatao and Fan, Angela},
        journal={arXiv preprint arXiv:2008.00401},
        year={2020}
    }
    ```
2. The transformer model published in [Beyond English-Centric Multilingual Machine Translation](https://arxiv.org/abs/2010.11125) by Fan et. al, which supports over 100 languages. You can cite it here:
   ```
   @misc{fan2020englishcentric,
        title={Beyond English-Centric Multilingual Machine Translation}, 
        author={Angela Fan and Shruti Bhosale and Holger Schwenk and Zhiyi Ma and Ahmed El-Kishky and Siddharth Goyal and Mandeep Baines and Onur Celebi and Guillaume Wenzek and Vishrav Chaudhary and Naman Goyal and Tom Birch and Vitaliy Liptchinsky and Sergey Edunov and Edouard Grave and Michael Auli and Armand Joulin},
        year={2020},
        eprint={2010.11125},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
   ```

`dlt` is a wrapper with useful `utils` to save you time. For huggingface's `transformers`, the following snippet is shown as an example:
```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# translate Hindi to French
tokenizer.src_lang = "hi_IN"
encoded_hi = tokenizer(article_hi, return_tensors="pt")
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"])
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire en Syria."

# translate Arabic to English
tokenizer.src_lang = "ar_AR"
encoded_ar = tokenizer(article_ar, return_tensors="pt")
generated_tokens = model.generate(**encoded_ar, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "The Secretary-General of the United Nations says there is no military solution in Syria."
```

With `dlt`, you can run:
```python
import dl_translate as dlt

article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."

mt = dlt.TranslationModel()
translated_fr = mt.translate(article_hi, source=dlt.lang.HINDI, target=dlt.lang.FRENCH)
translated_en = mt.translate(article_ar, source=dlt.lang.ARABIC, target=dlt.lang.ENGLISH)
```

Notice you don't have to think about tokenizers, condition generation, pretrained models, and regional codes; you can just tell the model what to translate!

If you are experienced with `huggingface`'s ecosystem, then you should be familiar enough with the example above that you wouldn't need this library. However, if you've never heard of huggingface or mBART, then I hope using this library will give you enough motivation to [learn more about them](https://github.com/huggingface/transformers) :)
