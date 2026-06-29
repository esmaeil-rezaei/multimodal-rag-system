# Contributing

Thanks for your interest in improving this project. This guide covers local
setup, development workflow, and the checks your changes need to pass.

## Development setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install runtime + development dependencies
pip install -r requirements.txt -r requirements-dev.txt

# 3. Install pre-commit hooks
pre-commit install

# 4. Copy the example environment file and fill in your own credentials
cp .env.example .env
```

## Running the stack

The app depends on Qdrant, Elasticsearch, Redis, and (optionally) Neo4j.
The simplest way to bring everything up locally is Docker Compose:

```bash
docker compose up --build
```

This starts the API on `http://localhost:8000` (see `/health` for a liveness
check) along with Qdrant (6333), Elasticsearch (9200), Redis (6379), and
Neo4j (7474/7687).

To run the API directly against locally-running dependencies instead:

```bash
uvicorn app.main:app --reload
```

## Development workflow

1. Create a branch off `main` for your change.
2. Make your changes, keeping commits focused and descriptive.
3. Add or update tests for any behavior change.
4. Run the checks below before opening a pull request.
5. Open a PR against `main`. CI will run lint, format check, type-check, and
   the unit test suite automatically.

## Checks

All tooling configuration lives in `pyproject.toml` (ruff, black, mypy) and
`pytest.ini` (pytest).

```bash
# Lint
ruff check .

# Format (check only)
black --check .

# Format (apply)
black .

# Type-check
mypy src app

# Unit tests
pytest -q
```

Or run all of the above (plus a few extra hygiene checks) at once via
pre-commit:

```bash
pre-commit run --all-files
```

## Tests

Unit tests live under `tests/unit/`. Tests that require the full ML
dependency stack (`sentence-transformers`, `scikit-learn`, etc.) are gated
with `pytest.importorskip(...)` so the suite still runs in lightweight
environments — they're skipped there but run in CI and any environment with
the full `requirements.txt` installed.

The 7-layer offline evaluation suite (`scripts/evaluate_offline.py` and
`tests/golden_*.json`) is separate from the unit test suite — see the
"Offline Evaluation" section of the README for how to run it.

## Security

Never commit `.env` or any file containing real credentials. `.env.example`
documents every variable the app expects. Tests must never load the real
`.env` file — `tests/conftest.py` disables `.env` loading and supplies dummy
secrets for the test session.

## Commit style

This project loosely follows [Conventional Commits](https://www.conventionalcommits.org/)
(e.g., `feat: ...`, `fix: ...`, `docs: ...`, `test: ...`, `chore: ...`) for
commit messages.
