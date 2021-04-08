import dl_translate as dlt


def test_save():
    mt = dlt.TranslationModel()
    mt.save_obj('saved_model')

def test_load():
    mt = dlt.TranslationModel.load_obj('saved_model')