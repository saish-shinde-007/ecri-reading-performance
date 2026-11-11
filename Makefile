.PHONY: install data synth check eda baseline test lint verify all clean

PY := .venv/bin/python

install:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev]"

data:
	@mkdir -p data/raw
	curl -fSL -o data/raw/ECRIDataSet.csv https://raw.githubusercontent.com/ecri-data/ECRI-Dataset/main/ECRI/ECRIDataSet.csv
	curl -fSL -o data/raw/ECRIRawData.csv https://raw.githubusercontent.com/ecri-data/ECRI-Dataset/main/ECRI/ECRIRawData.csv
	curl -fSL -o data/raw/MetaData.csv    https://raw.githubusercontent.com/ecri-data/ECRI-Dataset/main/ECRI/MetaData.csv
	@echo "ECRI data is CC BY-NC-SA 4.0 — cite Shangguan et al. (2024). See docs/data.md."

check:
	$(PY) -m ecri.cli check

synth:
	$(PY) -m ecri.cli synth

eda:
	$(PY) -m ecri.cli eda

baseline:
	$(PY) -m ecri.cli baseline --repeats 8

test:
	$(PY) -m pytest --cov --cov-report=term-missing --cov-fail-under=90

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

verify:
	$(PY) tools/verify_target_rule.py

all: check eda baseline

clean:
	rm -rf reports/figures/*.png reports/*.md data/interim/* data/processed/*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
