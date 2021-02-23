# DL Translate

*A deep learning-based translation library built on Huggingface `transformers` and Facebook's `mBART-Large`*

## Quickstart

Install the library with pip:
```
pip install dl-translate
```

Translate some text:

```python
import dl_translate as dlt

model = dlt.TranslationModel()  # Slow when you load it for the first time

text_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
model.translate(text_hi, source=dlt.lang.HINDI, target=dlt.lang.ENGLISH)
```

Above, you can see that `dlt.lang` contains variables representing each of the 50 available languages with auto-complete support. Alternatively, you can specify the language (e.g. "Arabic") or the language code (e.g. "fr_XX" for French):
```python
text_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."
model.translate(text_ar, source="Arabic", target="fr_XX")
```

You can use `dlt.utils` to find out which languages and codes are available:
```python
print(dlt.utils.available_languages())  # All languages that you can use
print(dlt.utils.available_codes())  # Code corresponding to each language accepted
print(dlt.utils.get_lang_code_map())  # Dictionary of lang -> code
```

## Usage

### Selecting a device

When you load the model, you can specify the device:
```python
model = dlt.TranslationModel(device="auto")
```

By default, the value will be `device="auto"`, which means it will use a GPU if possible. You can also explicitly set `device="cpu"` or `device="gpu"`, or some other strings accepted by [`torch.device()`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device). __In general, it is recommend to use a GPU if you want a reasonable processing time.__


### Loading from a path

By default, `dlt.TranslationModel` will download the model from the [huggingface repo](https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt) and cache it. However, you are free to load from a path:
```python
model = dlt.TranslationModel("/path/to/your/model/directory/")
```
Make sure that your tokenizer is also stored in the same directory if you use this approach.

### Using a different model

You can also choose another model that has [a similar format](https://huggingface.co/models?filter=mbart-50), e.g.
```python
model = dlt.TranslationModel("facebook/mbart-large-50-one-to-many-mmt")
```
Note that the available languages will change if you do this, so you will not be able to leverage `dlt.lang` or `dlt.utils`.

### Breaking down into sentences

It is not recommended to use extremely long texts as it takes more time to process. Instead, you can try to break them down into sentences with the help of `nltk`. First install the library with `pip install nltk`, then run:
```python
import nltk

nltk.load("punkt")

text = "Mr. Smith went to his favorite cafe. There, he met his friend Dr. Doe."
sents = nltk.tokenize.sent_tokenize(text, "english")  # don't use dlt.lang.ENGLISH
" ".join(model.translate(sents, source=dlt.lang.FRENCH, target=dlt.lang.FRENCH))
```


## Advanced

When initializing `model`, you can pass in arguments for the underlying BART model and tokenizer (which will respectively be passed to `MBartForConditionalGeneration.from_pretrained` and `MBart50TokenizerFast.from_pretrained`):

```python
model = dlt.TranslationModel(
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

You can access the underlying `transformers` model and `tokenizer`:
```python
bart = model.get_transformers_model()
tokenizer = model.get_tokenizer()
```

See the [huggingface docs](https://huggingface.co/transformers/master/model_doc/mbart.html) for more information.



## Acknowledgement

`dl-translate` is built on top of Huggingface's implementation of multilingual BART finetuned on many-to-many translation of over 50 languages, which is [documented here](https://huggingface.co/transformers/master/model_doc/mbart.html). The original paper was written by Tang et. al from Facebook AI Research; you can [find it here](https://arxiv.org/pdf/2008.00401.pdf) and cite it using the following:
```
@article{tang2020multilingual,
  title={Multilingual translation with extensible multilingual pretraining and finetuning},
  author={Tang, Yuqing and Tran, Chau and Li, Xian and Chen, Peng-Jen and Goyal, Naman and Chaudhary, Vishrav and Gu, Jiatao and Fan, Angela},
  journal={arXiv preprint arXiv:2008.00401},
  year={2020}
}
```

`dlt` is a thing wrapper with useful utils to save you time. For huggingface's `transformers`, the following snippet is shown as an example:
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

model = dlt.TranslationModel()
translated_fr = model.translate(article_hi, source=dlt.lang.HINDI, target=dlt.lang.FRENCH)
translated_en = model.translate(article_ar, source=dlt.lang.ARABIC, target=dlt.lang.ENGLISH)
```

If you are experienced with `huggingface`'s ecosystem, then you should be familiar enough with the example above that you wouldn't need this library. However, if you've never heard of huggingface or mBART, then I hope using this library will give you enough motivation to learn more about [them](https://github.com/huggingface/transformers) :)