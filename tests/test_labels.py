import numpy as np
import pandas as pd
import pytest

from ecri import labels


def _planted(n: int = 240):
    """Two time points per student, with each label set by a known gain cut."""
    rng = np.random.default_rng(0)
    stu = np.arange(1, n + 1)
    wid0, wda0 = rng.integers(0, 30, n), rng.integers(0, 15, n)
    wid_gain = np.resize(np.arange(0, 50), n)  # every gain 0..49 occurs
    wda_gain = np.resize(np.arange(0, 25), n)
    longitudinal = pd.DataFrame(
        {
            "StuID": np.repeat(stu, 2),
            "t": np.tile([0.0, 1.0], n),
            "RMwidRS": np.column_stack([wid0, wid0 + wid_gain]).ravel(),
            "RMwdaRS": np.column_stack([wda0, wda0 + wda_gain]).ravel(),
        }
    )
    analysis = pd.DataFrame(
        {
            "StuID": stu,
            "RMwidRS_target": (wid_gain >= 27).astype(float),
            "RMwdaRS_target": (wda_gain >= 11).astype(float),
        }
    )
    return analysis, longitudinal


def test_planted_thresholds_are_recovered_exactly():
    rules = labels.derive_rules(*_planted())
    assert rules["RMwidRS_target"]["threshold"] == 27
    assert rules["RMwdaRS_target"]["threshold"] == 11
    assert all(r["accuracy"] == 1.0 and r["n"] == 240 for r in rules.values())


def test_a_noisy_label_is_reported_as_inexact():
    analysis, longitudinal = _planted()
    analysis.loc[analysis.index[:12], "RMwidRS_target"] = 1 - analysis["RMwidRS_target"][:12]
    rule = labels.derive_rules(analysis, longitudinal)["RMwidRS_target"]
    assert rule["accuracy"] < 1.0


def test_unlabelled_students_are_left_out():
    analysis, longitudinal = _planted()
    analysis.loc[analysis.index[:40], ["RMwidRS_target", "RMwdaRS_target"]] = np.nan
    assert labels.derive_rules(analysis, longitudinal)["RMwidRS_target"]["n"] == 200


def test_missing_time_point_is_an_error():
    analysis, longitudinal = _planted()
    with pytest.raises(ValueError, match="not both present"):
        labels.derive_rules(analysis, longitudinal[longitudinal["t"] == 0.0])
