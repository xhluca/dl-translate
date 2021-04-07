import os
import json

from jinja2 import Template


def load_json(name):
    filepath = os.path.join(os.path.dirname(__file__), "langs_coverage", f"{name}.json")
    return json.loads(open(filepath).read())


template_values = {}
for name in ["m2m100", "mbart50"]:
    content = ""
    di = load_json(name)

    for key, val in di.items():
        content += f"- {key} ({val})\n"

    template_values[name] = content


template_path = os.path.join(
    os.path.dirname(__file__), "templates", "available_languages.md.jinja2"
)
save_path = os.path.join(
    os.path.dirname(__file__), "..", "docs", "available_languages.md"
)

with open(template_path) as f:
    template = Template(f.read())

rendered = template.render(template_values)

with open(save_path, "w") as f:
    f.write(rendered)
