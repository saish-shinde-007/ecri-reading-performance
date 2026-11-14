import numpy as np
import pytest

from ecri import evaluate, split

FAST = ["logistic", "majority", "tree_depth3"]


@pytest.fixture(scope="module")
def cv_run(prepared):
    return evaluate.run_models(prepared, "RMwidRS_target", FAST, repeats=2, n_folds=4, seed=3)


def test_fold_plan_covers_every_row_exactly_once_per_repeat(prepared):
    X, y, groups = split.features_and_target(prepared, "RMwidRS_target")
    plan = evaluate.make_fold_plan(X, y, groups, n_folds=4, repeats=3, seed=1)
    assert plan.n_repeats == 3 and plan.n_folds == 4
    for folds in plan.repeats:
        seen = np.concatenate([test for _, test in folds])
        assert sorted(seen.tolist()) == list(range(len(X)))


def test_school_fold_plan_never_splits_a_school(prepared):
    X, y, groups = split.features_and_target(prepared, "RMwidRS_target")
    plan = evaluate.make_fold_plan(X, y, groups, strategy="school", n_folds=4, repeats=2)
    for folds in plan.repeats:
        for train, test in folds:
            assert not set(groups.iloc[train]) & set(groups.iloc[test])


def test_repeats_must_be_positive(prepared):
    X, y, groups = split.features_and_target(prepared, "RMwidRS_target")
    with pytest.raises(ValueError, match="at least 1"):
        evaluate.make_fold_plan(X, y, groups, repeats=0)


def test_every_row_gets_an_out_of_fold_score(cv_run):
    results, plan = cv_run
    for result in results.values():
        assert result.oof_scores.shape == (plan.n_repeats, plan.n_rows)
        assert not np.isnan(result.oof_scores).any()


def test_majority_is_chance_and_logistic_beats_it(cv_run):
    results, _ = cv_run
    assert results["majority"].mean() == pytest.approx(0.5)
    assert results["logistic"].mean() > 0.5


def test_results_table_sorted_by_auc(cv_run):
    table = evaluate.results_table(cv_run[0])
    assert list(table["auc_mean"]) == sorted(table["auc_mean"], reverse=True)
    assert set(table.index) == set(FAST)


def test_a_model_compared_with_itself_differs_by_zero(cv_run):
    results, _ = cv_run
    c = evaluate.paired_bootstrap(results["logistic"], results["logistic"], n_boot=200)
    assert c.delta == pytest.approx(0.0)
    assert not c.decisive


def test_a_real_gap_is_reported_as_decisive(cv_run):
    results, _ = cv_run
    c = evaluate.paired_bootstrap(results["logistic"], results["majority"], n_boot=400)
    assert c.better == "logistic"
    assert c.delta > 0 and c.decisive
    assert "clears noise" in c.summary()


def test_comparison_direction_is_normalised(cv_run):
    results, _ = cv_run
    forward = evaluate.paired_bootstrap(results["logistic"], results["majority"], n_boot=200)
    reverse = evaluate.paired_bootstrap(results["majority"], results["logistic"], n_boot=200)
    assert forward.better == reverse.better == "logistic"
    assert forward.delta == pytest.approx(reverse.delta)


def test_compare_to_reference_excludes_the_reference(cv_run):
    comparisons = evaluate.compare_to_reference(cv_run[0], "logistic", n_boot=200)
    assert len(comparisons) == len(FAST) - 1
    assert all("logistic" in (c.better, c.worse) for c in comparisons)


def test_unknown_reference_rejected(cv_run):
    with pytest.raises(ValueError, match="was not cross-validated"):
        evaluate.compare_to_reference(cv_run[0], "random_forest")


def test_mismatched_rows_refuse_to_compare(cv_run, prepared):
    results, _ = cv_run
    other, _ = evaluate.run_models(
        prepared.iloc[:-30], "RMwidRS_target", ["logistic"], repeats=1, n_folds=4
    )
    with pytest.raises(ValueError, match="different rows"):
        evaluate.paired_bootstrap(results["logistic"], other["logistic"], n_boot=50)


def test_label_renames_the_result(prepared):
    X, y, groups = split.features_and_target(prepared, "RMwidRS_target")
    plan = evaluate.make_fold_plan(X, y, groups, n_folds=4, repeats=1)
    result = evaluate.cross_validate(X, y, plan, "logistic", label="feature_set_a")
    assert result.name == "feature_set_a"


def test_imputation_comparison_runs_every_variant(prepared):
    results, _ = evaluate.compare_imputation(prepared, "RMwidRS_target", repeats=1, n_folds=4)
    assert set(results) == set(evaluate.IMPUTATION_VARIANTS)


def test_imputation_comparison_adds_native_nan_only_where_supported(prepared):
    capable, _ = evaluate.compare_imputation(
        prepared, "RMwidRS_target", base="hist_gradient_boosting", repeats=1, n_folds=4
    )
    assert "native_nan" in capable
    incapable, _ = evaluate.compare_imputation(
        prepared, "RMwidRS_target", base="logistic", repeats=1, n_folds=4
    )
    assert "native_nan" not in incapable


def test_oof_frame_carries_slice_columns(cv_run, prepared):
    results, _ = cv_run
    frame = evaluate.oof_frame(results["logistic"], prepared, "RMwidRS_target")
    assert {"y_true", "score", "SchlID", "LEP", "SpEd"} <= set(frame.columns)
    assert len(frame) == len(prepared)
