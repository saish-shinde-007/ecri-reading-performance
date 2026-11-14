import numpy as np
import pytest

from ecri import calibrate


@pytest.fixture()
def scored():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    scores = np.clip(0.5 + 0.18 * (y - 0.5) + rng.normal(0, 0.2, 600), 0.01, 0.99)
    return y, scores


def test_three_calibrators_are_reported(scored):
    results = calibrate.fit_calibrators(*scored)
    assert set(results) == {"raw", "platt", "isotonic"}
    for r in results.values():
        assert 0.0 <= r.brier <= 1.0
        assert r.scores.shape == scored[1].shape


def test_threshold_minimises_the_stated_cost(scored):
    y, scores = scored
    best = calibrate.choose_threshold(y, scores, cost_ratio=2.0)
    for t in np.linspace(0.05, 0.95, 19):
        flagged = scores < t
        cost = (2.0 * ((y == 0) & ~flagged).sum() + ((y == 1) & flagged).sum()) / len(y)
        assert best.expected_cost <= cost + 1e-12


def test_a_costlier_miss_flags_at_least_as_many_students(scored):
    """F11's mechanism: raising the cost of a missed at-risk child can only push
    the threshold up and the flag rate up."""
    y, scores = scored
    rates = [
        calibrate.choose_threshold(y, scores, cost_ratio=r).flagged_rate for r in (1, 2, 3, 5, 10)
    ]
    assert rates == sorted(rates)


def test_cost_sweep_has_one_row_per_ratio(scored):
    sweep = calibrate.cost_sweep(*scored, ratios=(1, 3))
    assert list(sweep.index) == [1, 3]
    assert {"threshold", "flagged_rate", "recall_at_risk"} <= set(sweep.columns)


def test_non_positive_cost_ratio_rejected(scored):
    with pytest.raises(ValueError, match="positive"):
        calibrate.choose_threshold(*scored, cost_ratio=0)


def test_summary_names_the_cost_assumption(scored):
    text = calibrate.choose_threshold(*scored, cost_ratio=3).summary()
    assert "3:1" in text and "flags" in text


def test_reliability_figure_is_written(scored, tmp_path, monkeypatch):
    from ecri import paths

    monkeypatch.setattr(paths, "FIGURES", tmp_path)
    monkeypatch.setattr(paths, "WRITABLE", (tmp_path,))
    results = calibrate.fit_calibrators(*scored)
    out = calibrate.plot_reliability(scored[0], results, "unit")
    assert out.endswith("calibration_unit.png")
    assert (tmp_path / "calibration_unit.png").is_file()
