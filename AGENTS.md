# Repository Guidelines

## Project Structure & Module Organization
Core logic lives in `src/recsys_lite/`, split into focused modules: `api/` for FastAPI services, `cli/` for Typer commands, `ingest/` for data loaders, `models/` for recommenders, and `utils/` for shared helpers. Tests are organized under `tests/` (unit and service), with supporting scenarios in `test_scripts/` and fixtures in `test_data/`. Front-end assets sit in `widget/`, Docker orchestration in `docker/`, and generated artifacts (models, logs, SBOMs) land in the respective top-level directories.

## Build, Test, and Development Commands
Install dependencies with `poetry install` and activate tools through `poetry run ...`. Typical workflow:
- `poetry run recsys-lite ingest --events data/sample_data/events.parquet --items data/sample_data/items.csv` prepares demo data.
- `poetry run recsys-lite train ease` trains the default EASE-R model.
- `poetry run recsys-lite serve` or `poetry run python run_api.py` starts the API locally on port 8000.
- `make run` spins up the Docker development stack; `make run-prod` mirrors production compose settings.
- `make test` executes the containerised test suite; `make lint` runs linting targets; local quick runs can use `poetry run pytest`.

## Coding Style & Naming Conventions
We target Python 3.11 with 4-space indentation and keep lines ≤120 characters as enforced by Black and Ruff. Format code via `poetry run black .` and `poetry run isort .`; lint with `poetry run ruff check .`. Use explicit type hints—`poetry run mypy src` backs CI checks. Prefer snake_case for modules, functions, and CLI commands, PascalCase for classes, and keep configuration constants uppercase. Group FastAPI routes by feature in `src/recsys_lite/api` and place shared fixtures inside `src/recsys_lite/testing`.

## Testing Guidelines
Unit tests live beside their domains in `tests/`, while end-to-end coverage sits in `test_acceptance.py` and `test_scripts/`. Add new tests as `test_<feature>.py` and rely on `pytest` fixtures from `tests/conftest.py`. Run `poetry run pytest --cov=src/recsys_lite --cov-report=term-missing` to confirm coverage before opening a PR. Use sample payloads in `test_data/` and the lightweight CLI commands above to regenerate artifacts when updating golden files.

## Commit & Pull Request Guidelines
Follow the existing Conventional-Commit-inspired style: `fix:`, `chore:`, and `Refactor:` prefixes dominate the history (e.g., `fix: update dependencies to address security vulnerabilities`). Keep subjects imperative and under 72 characters. Each PR should link relevant issues, summarise behavioural changes, note impacts on training artifacts, and paste the test command outputs you ran. Include API or widget screenshots when UI behaviour changes and flag any migrations or data schema updates for reviewers.
