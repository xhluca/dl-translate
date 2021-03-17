# Contributing to the project

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
mkdocs serve
```