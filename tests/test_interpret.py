import pandas as pd
import pytest

from ecri import evaluate, interpret, split


@pytest.fixture(scope="module")
def importance(prepared):
    X, y, groups = split.features_and_target(prepared, "RMwidRS_target")
    plan = evaluate.make_fold_plan(X, y, groups, n_folds=4, repeats=1, seed=2)
    return interpret.permutation_table(X, y, plan, "logistic", n_repeats=3)


def test_one_row_per_raw_feature(importance, prepared):
    X, _, _ = split.features_and_target(prepared, "RMwidRS_target")
    assert set(importance.index) == set(X.columns)


def test_sorted_by_mean_drop(importance):
    assert list(importance["auc_drop_mean"]) == sorted(importance["auc_drop_mean"], reverse=True)


def test_meaningful_keeps_only_features_above_their_noise():
    table = pd.DataFrame(
        {
            "auc_drop_mean": [0.10, 0.02, 0.01],
            "auc_drop_sd": [0.02, 0.05, 0.001],
            "n_folds": [5, 5, 5],
        },
        index=["strong", "noisy", "small_but_steady"],
    )
    assert list(interpret.meaningful(table).index) == ["strong", "small_but_steady"]


def test_agreement_is_one_for_identical_rankings(importance):
    assert interpret.agreement(importance, importance) == pytest.approx(1.0)


def test_agreement_is_nan_with_too_little_overlap():
    a = pd.DataFrame({"auc_drop_mean": [0.1, 0.2]}, index=["x", "y"])
    assert pd.isna(interpret.agreement(a, a))
