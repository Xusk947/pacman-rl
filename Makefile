.PHONY: help venv install run smoke lint fmt clean

PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PY := $(VENV_DIR)/bin/python
PIP := $(VENV_PY) -m pip

help:
	@echo "Targets:"
	@echo "  make venv        Create virtual environment in $(VENV_DIR)"
	@echo "  make install     Install project in editable mode (and deps)"
	@echo "  make run         Run training (GPU by default)"
	@echo "  make smoke       Run a tiny smoke training run"
	@echo "  make clean       Remove run artifacts and caches"

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install -U pip

install: venv
	$(PIP) install -e .

run: install
	$(VENV_PY) -m pacman_rl.train --layout-dir layouts --device cuda --batch-size 256 --updates 200 --report-every 50 --telegram 

smoke: install
	$(VENV_PY) scripts/smoke_train.py

clean:
	rm -rf runs __pycache__ .pytest_cache .mypy_cache .ruff_cache
