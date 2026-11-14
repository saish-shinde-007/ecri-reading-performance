.PHONY: install data synth check eda baseline cv impute importance calibrate school model rank-demo test lint verify all clean

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

# Phase 2. Every comparison is repeated CV on shared folds (docs/findings.md F8).
cv:
	$(PY) -m ecri.cli cv --repeats 5

impute:
	$(PY) -m ecri.cli impute --repeats 5

importance:
	$(PY) -m ecri.cli importance --models logistic,random_forest --repeats 3

calibrate:
	$(PY) -m ecri.cli calibrate --repeats 5

school:
	$(PY) -m ecri.cli school --repeats 10

# T27/T28. `make model` writes models/logistic-RMwidRS_target/, which the app loads.
model:
	$(PY) -m ecri.cli model train --repeats 5
	$(PY) -m ecri.cli model verify

rank-demo: model
	$(PY) -m ecri.cli synth --features-only --rows 300
	$(PY) -m ecri.cli rank data/interim/synthetic_rank_input.csv --budget 20

test:
	$(PY) -m pytest --cov --cov-report=term-missing --cov-fail-under=90

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

verify:
	$(PY) tools/verify_target_rule.py

all: check eda cv impute importance calibrate school

clean:
	rm -rf reports/figures/*.png reports/*.md data/interim/* data/processed/*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
