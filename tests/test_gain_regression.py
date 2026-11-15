"""Tests for raw gain regression and late-timepoint leakage guard (T30)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecri.gain_regression import (
    assert_no_late_timepoint_leakage,
    build_gain_target,
    evaluate_gain_regression,
    load_gain_target,
)


def _synthetic_longitudinal(n_students: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for sid in range(1, n_students + 1):
        t0 = 100.0 + rng.normal(0, 10)
        t05 = t0 + 14.0 + rng.normal(0, 4)
        t1 = t0 + (32.0 if sid % 2 == 0 else 18.0) + rng.normal(0, 3)
        for t_val, score in ((0.0, t0), (0.5, t05), (1.0, t1)):
            rows.append({"StuID": sid, "t": t_val, "RMwidRS": score})
    return pd.DataFrame(rows)


def test_build_gain_target_and_load(tmp_path: Path) -> None:
    longitudinal = _synthetic_longitudinal(20)
    analysis = pd.DataFrame({"StuID": list(range(1, 21)), "f0": np.arange(20, dtype=float)})
    gain = build_gain_target(analysis, longitudinal)
    assert gain.name == "RMwidRS_gain"
    assert len(gain) == 20

    csv_path = tmp_path / "ECRIRawData.csv"
    longitudinal.to_csv(csv_path, index=False)
    loaded = load_gain_target(analysis, longitudinal_path=csv_path)
    pd.testing.assert_series_equal(gain, loaded)


def test_no_late_timepoint_leakage() -> None:
    longitudinal = _synthetic_longitudinal(20)
    baseline = longitudinal.loc[longitudinal["t"] < 0.5, ["StuID", "RMwidRS"]].rename(
        columns={"RMwidRS": "RMwidRS_t0"}
    )
    assert_no_late_timepoint_leakage(baseline, longitudinal)

    perturbed = longitudinal.copy()
    perturbed.loc[perturbed["t"] >= 0.5, "RMwidRS"] = 9999.0
    baseline_after = perturbed.loc[perturbed["t"] < 0.5, ["StuID", "RMwidRS"]].rename(
        columns={"RMwidRS": "RMwidRS_t0"}
    )
    pd.testing.assert_frame_equal(baseline, baseline_after)

    leaky = baseline.copy()
    leaky["RMwidRS_t1"] = 1.0
    with pytest.raises(ValueError, match="Leaky"):
        assert_no_late_timepoint_leakage(leaky, longitudinal)


def test_evaluate_gain_regression() -> None:
    longitudinal = _synthetic_longitudinal(60)
    baseline = longitudinal.loc[longitudinal["t"] < 0.5, ["StuID", "RMwidRS"]].reset_index(
        drop=True
    )
    baseline["signal"] = [1.0 if sid % 2 == 0 else 0.0 for sid in baseline["StuID"]]
    X = baseline[["RMwidRS", "signal"]]
    y_gain = build_gain_target(baseline, longitudinal)

    res = evaluate_gain_regression(X, y_gain, n_splits=3, n_repeats=4, n_boot=100, seed=42)
    assert res.n_students == 60
    assert math.isfinite(res.rmse_mean)
    assert math.isfinite(res.mae_mean)
    assert math.isfinite(res.r2_mean)
    assert 0.0 <= res.thresholded_roc_auc_mean <= 1.0
    assert 0.0 <= res.classifier_roc_auc_mean <= 1.0
    assert res.ci_low <= res.ci_high
