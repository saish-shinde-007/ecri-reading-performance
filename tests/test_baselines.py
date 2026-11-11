import pytest

from ecri import metrics
from ecri.models import baselines


def test_every_baseline_runs(split_data):
    results = baselines.run_baselines(split_data, seed=3)
    assert {r.name for r in results} == set(baselines.BASELINES)
    for r in results:
        assert 0.0 <= r.metrics.roc_auc <= 1.0


def test_majority_baseline_is_exactly_chance(split_data):
    results = {r.name: r for r in baselines.run_baselines(split_data, seed=3)}
    assert results["majority"].metrics.roc_auc == pytest.approx(0.5)
    assert results["majority"].metrics.recall_at_risk == 0.0


def test_logistic_beats_guessing(split_data):
    results = {r.name: r for r in baselines.run_baselines(split_data, seed=3)}
    assert results["logistic"].metrics.roc_auc > results["stratified"].metrics.roc_auc


def test_no_baseline_looks_like_a_leak(split_data):
    """A real result here is ~0.66 AUC (F7). Anything near-perfect means the
    guard upstream stopped working."""
    for r in baselines.run_baselines(split_data, seed=3):
        assert r.metrics.roc_auc < metrics.SUSPICIOUS_AUC, r.name
        assert r.warning is None


def test_baseline_only_model_uses_one_column(split_data):
    pipeline = baselines.build_baseline("logistic_baseline_only", split_data.X_train)
    _, _, columns = pipeline.named_steps["pre"].transformers[0]
    assert columns == ["RMwidRS"]


def test_unknown_baseline_rejected(split_data):
    with pytest.raises(ValueError, match="unknown baseline"):
        baselines.build_baseline("transformer", split_data.X_train)


def test_results_table_is_sorted_by_auc(split_data):
    table = baselines.results_table(baselines.run_baselines(split_data, seed=3))
    assert list(table["roc_auc"]) == sorted(table["roc_auc"], reverse=True)
    assert "majority" in table.index
