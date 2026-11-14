import copy

import pytest

from ecri import reproduce


@pytest.fixture(scope="module")
def record():
    return reproduce.load_expected()


def test_the_record_uses_the_canonical_configuration(record):
    # Editing CANONICAL without re-recording would make every comparison void.
    assert record["config"] == reproduce.CANONICAL


def test_the_record_covers_every_canonical_model(record):
    assert set(record["model_comparison"]["models"]) == set(reproduce.CANONICAL_MODELS)
    vs = record["model_comparison"]["vs_reference"]
    assert set(vs) == set(reproduce.CANONICAL_MODELS) - {reproduce.CANONICAL["reference"]}


def test_the_record_pins_the_data_it_came_from(record):
    from ecri import sources

    assert record["data"]["commit"] == sources.COMMIT
    assert record["data"]["sha256"] == sources.SHA256


def test_the_label_rule_is_recorded_as_exact(record):
    rules = record["label_rule"]
    assert rules["RMwidRS_target"]["threshold"] == 27.0
    assert rules["RMwdaRS_target"]["threshold"] == 11.0
    assert all(r["accuracy"] == 1.0 and r["n"] == 946 for r in rules.values())


def test_the_readme_table_is_generated_from_the_record(record):
    """The README results table is never hand-typed: it must equal what
    `ecri readme` renders from results/expected.json, byte for byte."""
    current = reproduce.README_PATH.read_text()
    assert reproduce.render_readme(current, record) == current


def test_an_identical_run_passes(record):
    checks, unrecorded = reproduce.compare(copy.deepcopy(record), record)
    assert checks and all(c.ok for c in checks)
    assert unrecorded == []


def test_drift_inside_the_third_decimal_passes_and_beyond_fails(record):
    actual = copy.deepcopy(record)
    actual["model_comparison"]["models"]["logistic"]["auc_mean"] += 0.0004
    assert all(c.ok for c in reproduce.compare(actual, record)[0])

    actual["model_comparison"]["models"]["logistic"]["auc_mean"] += 0.0002
    failed = [c for c in reproduce.compare(actual, record)[0] if not c.ok]
    assert [c.key for c in failed] == ["model_comparison.models.logistic.auc_mean"]
    assert "FAIL" in failed[0].line()


def test_a_flipped_verdict_fails_even_when_numbers_are_close(record):
    actual = copy.deepcopy(record)
    vs = actual["model_comparison"]["vs_reference"]["random_forest"]
    vs["decisive"] = not vs["decisive"]
    failed = [c for c in reproduce.compare(actual, record)[0] if not c.ok]
    assert [c.key for c in failed] == ["model_comparison.vs_reference.random_forest.decisive"]


def test_a_missing_number_fails(record):
    actual = copy.deepcopy(record)
    del actual["deployed_model"]["auc_sd"]
    failed = [c for c in reproduce.compare(actual, record)[0] if not c.ok]
    assert [c.key for c in failed] == ["deployed_model.auc_sd"]


def test_new_numbers_are_reported_not_failed(record):
    actual = copy.deepcopy(record)
    actual["model_comparison"]["models"]["xgboost"] = {"auc_mean": 0.66}
    checks, unrecorded = reproduce.compare(actual, record)
    assert all(c.ok for c in checks)
    assert unrecorded == ["model_comparison.models.xgboost.auc_mean"]


def test_a_changed_configuration_is_refused(record):
    actual = copy.deepcopy(record)
    actual["config"]["repeats"] = 3
    with pytest.raises(reproduce.ReproductionError, match="not comparable"):
        reproduce.compare(actual, record)


def test_the_gate_refuses_to_run_on_the_wrong_data(tmp_path, monkeypatch):
    from ecri import paths

    monkeypatch.setattr(paths, "RAW", tmp_path)
    with pytest.raises(reproduce.ReproductionError, match="make data"):
        reproduce.run()


def test_readme_block_orders_by_auc_and_marks_verdicts(record):
    block = reproduce.readme_block(record)
    rows = [line for line in block.splitlines() if line.startswith("| ") and "---" not in line]
    names = [row.split("|")[1].strip().strip("*") for row in rows[1:]]
    assert names[0] == "Logistic regression"
    aucs = [float(row.split("|")[2].strip().strip("*").split(" ± ")[0]) for row in rows[1:]]
    assert aucs == sorted(aucs, reverse=True)
    assert "**worse**" in block and "within noise" in block
    assert "−" in block and "-0." not in block  # true minus signs, never hyphens


def test_render_readme_touches_only_the_block(record):
    text = "intro\n" + reproduce.README_START + "\nstale\n" + reproduce.README_END + "\noutro\n"
    out = reproduce.render_readme(text, record)
    assert out.startswith("intro\n") and out.endswith("\noutro\n")
    assert "stale" not in out


def test_render_readme_needs_its_markers(record):
    with pytest.raises(reproduce.ReproductionError, match="results:start"):
        reproduce.render_readme("no markers here", record)
