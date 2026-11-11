import numpy as np
import pandas as pd
import pytest

from ecri import features, schema


def test_output_is_finite_and_dense(split_data):
    pre = features.build_preprocessor(split_data.X_train)
    out = pre.fit_transform(split_data.X_train)
    assert np.isfinite(out).all()
    assert out.shape[0] == split_data.n_train


def test_missingness_indicators_are_emitted(split_data):
    pre = features.build_preprocessor(split_data.X_train)
    pre.fit(split_data.X_train)
    names = features.feature_names(pre)
    assert any("missing" in n for n in names), names[:10]


def test_fit_on_train_only_does_not_see_test(split_data):
    """Scaling statistics must come from training rows alone; otherwise test
    information crosses the split before a model ever sees it."""
    pre = features.build_preprocessor(split_data.X_train)
    train_out = pre.fit_transform(split_data.X_train)
    # Train columns standardise to ~0 mean; test columns need not.
    numeric_width = len(schema.roles_for(list(split_data.X_train.columns)).numeric)
    assert np.abs(train_out[:, :numeric_width].mean(axis=0)).max() < 1e-8


def test_transform_is_stable_across_calls(split_data):
    pre = features.build_preprocessor(split_data.X_train)
    pre.fit(split_data.X_train)
    a = pre.transform(split_data.X_test)
    b = pre.transform(split_data.X_test)
    assert np.allclose(a, b)


def test_unseen_category_does_not_explode(split_data):
    pre = features.build_preprocessor(split_data.X_train)
    pre.fit(split_data.X_train)
    weird = split_data.X_test.copy()
    weird.loc[:, "Tier"] = 99.0
    assert np.isfinite(pre.transform(weird)).all()


def test_columns_outside_the_contract_are_rejected(split_data):
    with pytest.raises(ValueError, match="outside the contract"):
        features.build_preprocessor(split_data.X_train.assign(mystery=1.0))


def test_empty_feature_set_rejected():
    with pytest.raises(ValueError, match="no contract features"):
        features.build_preprocessor(pd.DataFrame(index=range(3)))


def test_all_missing_column_survives_imputation(split_data):
    X = split_data.X_train.copy()
    X.loc[:, "ORFwc"] = np.nan
    pre = features.build_preprocessor(X)
    assert np.isfinite(pre.fit_transform(X)).all()
