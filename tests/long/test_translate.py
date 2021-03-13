import dl_translate as dlt


def test_translate():
    mt = dlt.TranslationModel()

    msg_en = "Hello everyone, how are you?"

    assert (
        mt.translate(msg_en, source="English", target="Spanish")
        == "Hola a todos, ¿cómo va?"
    )

    fr_1 = mt.translate(msg_en, source="English", target="French")
    ch = mt.translate(msg_en, source="English", target="Chinese")
    fr_2 = mt.translate([msg_en], source="English", target="French")

    assert fr_1 == fr_2[0]
    assert ch != fr_1
