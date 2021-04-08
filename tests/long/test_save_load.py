import os

import dl_translate as dlt


def test_save():
    mt = dlt.TranslationModel()
    mt.save_obj("saved_model")
    assert os.path.exists("saved_model/weights.pt")
    assert os.path.exists("saved_model/tokenizer_config.json")


def test_load():
    mt = dlt.TranslationModel.load_obj("saved_model")
    assert isinstance(mt, dlt.TranslationModel)
