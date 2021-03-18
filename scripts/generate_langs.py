import json
import os


def load_json(name):
    filepath = os.path.join(os.path.dirname(__file__), "langs_coverage", f"{name}.json")
    return json.loads(open(filepath).read())


auto_gen_comment = f"# Auto-generated. Do not modify, use {__file__} instead.\n"

name2json = {}

for name in ["m2m100", "mbart50"]:
    name2json[name] = lang2code = load_json(name)

    with open(f"./dl_translate/lang/{name}.py", "w") as f:
        f.write(auto_gen_comment)
        for lang, code in lang2code.items():
            f.write(f'{lang.upper().replace(" ", "_")} = "{lang}"\n')


with open("./dl_translate/_pairs.py", "w") as f:
    f.write(auto_gen_comment)

    for name, lang2code in name2json.items():
        f.write(f"_PAIRS_{name.upper()} = {tuple(lang2code.items())}\n")
