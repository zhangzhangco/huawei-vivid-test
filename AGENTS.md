# Repository Guidelines

## Project Structure & Module Organization
Core logic lives in `src/`, with `src/core/` providing Phoenix curve math, validation, smoothing, and export utilities, while `src/gradio_app.py` and `src/main.py` expose the UI and CLI entry points. Gradio configuration sits in `app.py` and `launch_gradio.py`. Shared presets and interface strings are stored under `config/`. Use `examples/` to stage reproducible workflows and keep large media assets outside the repo. All reference material, including API and developer docs, is in `docs/`. Automated checks and fixtures are collected in `tests/`, and `test_hdr_upload.py` guards the upload pipeline.

## Build, Test, and Development Commands
Set up dependencies with `python -m pip install -r requirements.txt` for runtime or `python -m pip install -e .[dev]` when contributing. Launch the interactive demo via `python launch_gradio.py` or `gradio app.py`. Run the full unit suite with `python -m pytest tests -q`, and execute the end-to-end validation flow using `python tests/run_validation_tests.py --test-type all --report-format both`. Regenerate golden datasets only when behavior changes (`python tests/run_validation_tests.py --generate-data`).

## Coding Style & Naming Conventions
Target Python 3.8+ and follow the existing four-space indentation, type hints, and docstring patterns in `src/core/`. Modules and functions use `snake_case`, public classes use `PascalCase`, and test names should mirror the module under test (`test_<module>.py::test_<behavior>`). Format code with `black .` and enforce linting with `flake8 src tests`. Prefer explicit imports from `src.core` to keep public APIs discoverable.

## Testing Guidelines
Unit tests live beside domain modules in `tests/`, exercising corner cases for PQ conversion, Phoenix calculations, and state handling. Add at least one focused pytest case per new branch or regression fix, using parametrization for numeric ranges. For integration work, extend `test_integration_final.py` or wire in new scenarios through `AutomatedRegressionTestSuite`. Keep validation fixtures current by running the regression runner after major changes and updating golden data in versioned directories. Aim to keep `pytest --maxfail=1 --cov=src` at or above the current coverage in CI.

## Commit & Pull Request Guidelines
Commits follow the existing convention of leading with an expressive emoji and a short, imperative summary (e.g., `🔧 修复曲线更新错误输出`). Group related changes logically; avoid mixing refactors with feature work. Pull requests should outline motivation, testing evidence (`pytest`, validation runner output), and any new configuration steps. Link to tracked issues when available and attach screenshots or curve plots when UI behavior shifts. Confirm that docs and examples reflect user-visible changes before requesting review.
