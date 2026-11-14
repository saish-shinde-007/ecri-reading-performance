.PHONY: install data synth check eda baseline cv impute importance calibrate school model rank-demo test lint verify reproduce readme release-check all clean

PY := .venv/bin/python

# Exactly the versions in uv.lock; --locked fails if pyproject and the lock disagree.
install:
	uv sync --locked --extra dev

# Pinned to a commit of the dataset repo; every file is checked against its sha256.
data:
	$(PY) tools/fetch_data.py

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
	.venv/bin/ruff check src tests tools
	.venv/bin/ruff format --check src tests tools

verify:
	$(PY) tools/verify_target_rule.py

# T39: re-derive the headline numbers and compare them with results/expected.json.
reproduce:
	$(PY) -m ecri.cli reproduce

# Regenerate the README results table from results/expected.json.
readme:
	$(PY) -m ecri.cli readme

# Everything that must pass before a release tag — see RELEASING.md.
release-check: lint test
	$(PY) tools/fetch_data.py --check
	$(PY) tools/verify_target_rule.py
	$(PY) -m ecri.cli readme --check
	$(PY) -m ecri.cli reproduce

# Every report in reports/, plus the saved model, regenerated from scratch.
all: check eda baseline cv impute importance calibrate school model

clean:
	rm -rf reports/figures/*.png reports/*.md reports/*.csv data/interim/* data/processed/*
	rm -rf models/*/
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
