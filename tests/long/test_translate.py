import dl_translate as dlt


def test_translate():
    model = dlt.TranslationModel()

    msg_en = "Hello everyone, how are you?"

    assert (
        model.translate(msg_en, source="English", target="Spanish")
        == "Hola a todos, ¿cómo va?"
    )

    fr_1 = model.translate(msg_en, source="English", target="French")
    ch = model.translate(msg_en, source="English", target="Chinese")
    fr_2 = model.translate([msg_en], source="English", target="French")

    assert fr_1 == fr_2[0]
    assert ch != fr_1
