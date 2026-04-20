---
name: python-best-practices
description: Apply modern Python best practices (PEP 8, typing, errors, logging, testing, packaging). Use when writing or reviewing Python code, setting up project structure, or when the user asks for "best practices", "clean code", "PEP 8", "typing", "pytest", "ruff/black", or "pyproject.toml".
---

# Python Best Practices

This Skill guides you in writing maintainable, testable, secure Python code with modern tooling and conventions.

## Sources (primary)

- PEP 8 – Style Guide for Python Code: https://peps.python.org/pep-0008/

## Quick start

When asked to write or improve Python code:

1. Clarify constraints (Python version, runtime, performance needs, target OS, allowed dependencies).
2. Propose a minimal, idiomatic design (small modules, clear interfaces).
3. Implement with type hints, explicit error handling, and tests.
4. Keep I/O, logging, and configuration explicit and injectable.

## Instructions

### 1) Project structure

- Prefer a `src/` layout for libraries.
- Keep modules small and cohesive; avoid circular imports.
- Put configuration in one place (env vars + config file), avoid hidden globals.

Example:

```text
project/
  pyproject.toml
  src/
    package_name/
      __init__.py
      api.py
      core.py
  tests/
    test_core.py
```

### 2) Style and readability (PEP 8 aligned)

- Use 4 spaces, consistent naming (snake_case for functions/vars, CapWords for classes).
- Avoid overly long functions; prefer early returns.
- Keep imports grouped: stdlib, third-party, local.
- Avoid bare `except:`; catch specific exceptions and handle them intentionally.

### 3) Typing

- Use type hints on public APIs and non-trivial internal functions.
- Prefer explicit return types for functions that are reused.
- Use `dataclasses` for simple data containers.
- Keep types practical; don’t fight the type checker.

### 4) Errors and reliability

- Raise exceptions with actionable messages.
- Wrap errors with context at module boundaries (e.g., parsing, I/O, network).
- Avoid using exceptions for normal control flow.

### 5) Logging

- Use structured, minimal logging; do not log secrets or personal data.
- Prefer passing a logger (or using module-level logger) rather than printing.
- Log at boundaries: request handlers, job orchestration, external I/O.

### 6) Testing

- Prefer pytest-style tests if available; otherwise use `unittest`.
- Use table-driven tests for pure logic.
- Separate unit tests (fast) from integration tests (DB/network).

### 7) Security

- Avoid `eval`/`exec` on untrusted input.
- Validate and sanitize inputs at boundaries.
- Don’t hardcode secrets; use environment variables or secret managers.

### 8) Tooling recommendations (adapt to repo conventions)

- Formatter: `black` (or `ruff format`).
- Linter: `ruff`.
- Type checker: `mypy` or `pyright`.
- Tests: `pytest`.

If the repo already uses alternatives, follow existing tooling instead of introducing new ones.

## Output format

When applying this Skill, produce:

1. A short list of conventions you will follow (naming, imports, errors, testing).
2. Proposed module/file layout (if relevant).
3. The implementation (or review) with specific, actionable changes.
4. Minimal tests or verification steps consistent with the repo.
