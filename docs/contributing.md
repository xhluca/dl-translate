# Contributions

If you wish to contribute to the project, please do the following:
1. Verify if there's an existing similar issue.
2. If no issue exists, create it.
3. Once the contribution has been discussed inside the issue, fork this repo.
4. Before modifying any code, make sure to read the sections below.
5. Once you are done with your contribution, start a PR and tag a codeowner.


## Setup

To set up the development environment, clone the repo:

```bash
git clone https://github.com/xhlulu/dl-translate
cd dl-translate
```

Create a new venv and install the dev dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -e .[dev]
```

## Code linting

To ensure consistent and readable code, we use `black`. To run:

```bash
python black .
```

## Running tests

To run **all** the tests:
```bash
python -m pytest tests
```

For quick tests, run:
```bash
python -m pytest tests/fast
```

## Documentation

To re-generate the documentation after the source code was modified:
```bash
python scripts/render_references.py
```

To run the docs locally, run:
```
mkdocs serve -t material
```

Once ready, you can build it:
```
mkdocs build -t material
```

Or release it on GitHub Pages:
```
mkdocs gh-deploy -t material
```