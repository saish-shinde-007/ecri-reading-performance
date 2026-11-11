import math

import numpy as np
import pytest

from ecri import metrics


def test_perfect_predictions():
    m = metrics.evaluate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m.roc_auc == 1.0
    assert m.accuracy == 1.0
    assert m.recall_at_risk == 1.0
    assert (m.tn, m.fp, m.fn, m.tp) == (2, 0, 0, 2)


def test_inverted_predictions_score_zero_auc():
    m = metrics.evaluate([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert m.roc_auc == 0.0


def test_recall_at_risk_is_recall_on_the_negative_class():
    # Three at-risk students (label 0); the model catches two of them.
    m = metrics.evaluate([0, 0, 0, 1], [0.1, 0.2, 0.9, 0.9])
    assert m.recall_at_risk == pytest.approx(2 / 3)
    assert m.recall == 1.0


def test_single_class_gives_nan_not_zero():
    m = metrics.evaluate([1, 1, 1], [0.4, 0.6, 0.8])
    assert math.isnan(m.roc_auc)
    assert math.isnan(m.average_precision)


def test_threshold_is_respected():
    low = metrics.evaluate([0, 1], [0.4, 0.6], threshold=0.5)
    high = metrics.evaluate([0, 1], [0.4, 0.6], threshold=0.7)
    assert low.tp == 1
    assert high.tp == 0


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        metrics.evaluate([0, 1], [0.5])


def test_empty_input_rejected():
    with pytest.raises(ValueError, match="no rows"):
        metrics.evaluate([], [])


def test_leakage_warning_fires_above_the_ceiling():
    good = metrics.evaluate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics.leakage_warning(good) is not None
    assert "leakage" in metrics.leakage_warning(good)


def test_leakage_warning_silent_on_a_plausible_score():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    scores = np.clip(y * 0.12 + rng.normal(0.5, 0.25, 400), 0, 1)
    m = metrics.evaluate(y, scores)
    assert m.roc_auc < metrics.SUSPICIOUS_AUC
    assert metrics.leakage_warning(m) is None


def test_summary_and_dict_round_trip():
    m = metrics.evaluate([0, 1, 0, 1], [0.2, 0.7, 0.4, 0.9])
    assert "AUC=" in m.summary()
    assert m.as_dict()["n"] == 4
