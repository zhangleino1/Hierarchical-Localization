# Repository Guidelines

## Project Structure & Module Organization
`hloc/` is the main Python package. Top-level scripts such as `extract_features.py`, `match_features.py`, `reconstruction.py`, and `localize_sfm.py` implement the core pipeline, while `extractors/`, `matchers/`, `utils/`, and `pipelines/` hold model adapters, helpers, and dataset-specific entry points. Example notebooks live at the repo root (`demo.ipynb`, `pipeline_*.ipynb`). Static figures and design notes are in `doc/`. Keep benchmark inputs under `datasets/` and predefined retrieval pairs under `pairs/`. `third_party/` contains git submodules; treat it as vendored code unless a change must be upstreamed there.

## Build, Test, and Development Commands
Install locally with:
```bash
python -m pip install -e .
git submodule update --init --recursive
```
Key checks:
```bash
python -m flake8 hloc
python -m isort hloc *.ipynb --check-only --diff
python -m black hloc *.ipynb --check --diff
python -m compileall hloc
```
Run a pipeline module directly when validating behavior, for example:
```bash
python -m hloc.pipelines.Aachen.pipeline --outputs ./outputs/aachen
```
Use `docker build -t hloc:latest .` only when reproducing the containerized notebook workflow from `README.md`.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and a maximum line length of 88 characters. `flake8` is configured in `.flake8`; imports should stay `black`-compatible via `.isort.cfg`. Use `snake_case` for functions, modules, and local variables; keep class names in `PascalCase`. Match existing script-style module names in `hloc/` and favor explicit configuration dictionaries over hidden globals.

## Testing Guidelines
There is no dedicated `tests/` package or coverage gate in this repository. Contributors should treat linting plus targeted smoke tests as the baseline: run the lint commands above, import the package successfully, and execute the smallest relevant pipeline or script for the code you changed. For notebook changes, rerun the affected notebook cells end to end or provide an equivalent CLI reproduction.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects, often with Conventional Commit prefixes such as `feat:` and `docs:`. Keep commits focused and descriptive, for example `feat: add LightGlueStick matcher option`. Pull requests should summarize the pipeline or dataset impact, list validation commands, and link related issues or papers. Include screenshots only when notebook outputs, plots, or documentation visuals change.

## Data & Dependency Notes
Do not commit large datasets, generated models, or output folders. When a feature depends on `pycolmap`, verify compatibility with the version required in `requirements.txt` and surfaced in `hloc/__init__.py`.
