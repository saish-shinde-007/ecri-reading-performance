import pytest

from ecri import schema, split


def test_schools_are_disjoint(prepared):
    result = split.make_split(prepared, "RMwidRS_target", strategy="school", seed=1)
    train_schools = set(result.groups_train)
    X, _, groups = split.features_and_target(prepared, "RMwidRS_target")
    test_rows = len(X) - result.n_train
    assert test_rows == result.n_test
    # Reconstruct the held-out schools and confirm no overlap.
    all_schools = set(groups)
    assert train_schools < all_schools
    assert result.n_train + result.n_test == len(X)


def test_student_split_is_stratified(prepared):
    result = split.make_split(prepared, "RMwidRS_target", strategy="student", seed=1)
    assert abs(result.y_train.mean() - result.y_test.mean()) < 0.05


def test_target_never_reaches_the_features(split_data):
    assert "RMwidRS_target" not in split_data.X_train.columns


def test_identifiers_are_not_features(split_data):
    assert not set(schema.IDENTIFIERS) & set(split_data.X_train.columns)


def test_guard_rejects_the_target(prepared):
    X, y, _ = split.features_and_target(prepared, "RMwidRS_target")
    with pytest.raises(split.LeakageError, match="label-revealing"):
        split.assert_no_leakage(X.assign(RMwidRS_target=y), "RMwidRS_target")


def test_guard_rejects_post_baseline_columns(prepared):
    X, _, _ = split.features_and_target(prepared, "RMwidRS_target")
    with pytest.raises(split.LeakageError, match="label-revealing"):
        split.assert_no_leakage(X.assign(**{"RMwidRS_t1.0": 1.0}), "RMwidRS_target")


def test_guard_rejects_identifiers(prepared):
    X, _, _ = split.features_and_target(prepared, "RMwidRS_target")
    with pytest.raises(split.LeakageError, match="identifiers"):
        split.assert_no_leakage(X.assign(SchlID=1.0), "RMwidRS_target")


def test_cv_folds_keep_schools_whole(prepared):
    X, y, groups = split.features_and_target(prepared, "RMwidRS_target")
    folds = list(split.cv_folds(X, y, groups, strategy="school", n_folds=5))
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert not set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])


def test_cv_rejects_more_folds_than_schools(prepared):
    X, y, groups = split.features_and_target(prepared, "RMwidRS_target")
    with pytest.raises(ValueError, match="only"):
        list(split.cv_folds(X, y, groups, strategy="school", n_folds=99))


def test_school_split_needs_groups(prepared):
    with pytest.raises(ValueError, match="SchlID"):
        split.make_split(
            prepared.drop(columns=[schema.SCHOOL_ID]), "RMwidRS_target", strategy="school"
        )


def test_unknown_strategy_rejected(prepared):
    with pytest.raises(ValueError, match="unknown strategy"):
        split.make_split(prepared, "RMwidRS_target", strategy="kfold")
