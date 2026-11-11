.PHONY: install synth eda baseline test lint all clean

PY := .venv/bin/python

install:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev]"

synth:
	$(PY) -m ecri.cli synth

eda:
	$(PY) -m ecri.cli eda

baseline:
	$(PY) -m ecri.cli baseline

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

all: synth eda baseline

clean:
	rm -rf reports/figures/*.png reports/*.md data/interim/* data/processed/*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
