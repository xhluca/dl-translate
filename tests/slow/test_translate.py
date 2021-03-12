import dl_translate as dlt

model = dlt.TranslationModel()

msg_en = "Hello everyone, how are you today?"

print("Original Message:", msg_en)

print(
    "English -> French",
    model.translate(msg_en, source="English", target="French")
)

print(
    "English -> Chinese",
    model.translate(msg_en, source="English", target="Chinese")
)


print(
    "English -> French",
    model.translate([msg_en], source="English", target="French")
)