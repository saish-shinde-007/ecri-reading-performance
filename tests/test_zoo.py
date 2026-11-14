import numpy as np
import pytest

from ecri.features import NATIVE_NAN, PreprocessorConfig
from ecri.models import zoo


def test_every_registered_name_is_unique():
    names = [spec.name for spec in zoo.SPECS]
    assert len(names) == len(set(names))


def test_core_models_are_always_available():
    available = zoo.available_models()
    for name in ("logistic", "random_forest", "hist_gradient_boosting", "mlp", "majority"):
        assert name in available


def test_optional_models_hide_when_their_package_is_missing(monkeypatch):
    spec = zoo.ModelSpec(
        "phantom",
        "needs a package that is not there",
        "tree",
        lambda seed: None,
        requires="definitely_not_a_real_package",
    )
    assert not spec.available


def test_build_fits_and_predicts_probabilities(split_data):
    pipeline = zoo.build("logistic", split_data.X_train)
    pipeline.fit(split_data.X_train, split_data.y_train)
    proba = pipeline.predict_proba(split_data.X_test)
    assert proba.shape == (split_data.n_test, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_native_nan_model_receives_raw_nan(split_data):
    pipeline = zoo.build("hgb_native_nan", split_data.X_train)
    transformed = pipeline.named_steps["pre"].fit_transform(split_data.X_train)
    assert np.isnan(transformed).any(), "NaN should pass through to the estimator"


def test_baseline_only_model_is_restricted_to_one_column(split_data):
    spec = zoo.BY_NAME["logistic_baseline_only"]
    assert zoo.columns_for(spec, split_data.X_train) == ["RMwidRS"]


def test_unknown_model_rejected(split_data):
    with pytest.raises(ValueError, match="unknown model"):
        zoo.build("transformer", split_data.X_train)


def test_unavailable_model_refuses_to_build(split_data):
    spec = zoo.ModelSpec("phantom", "x", "tree", lambda seed: None, requires="no_such_pkg_xyz")
    with pytest.raises(ImportError, match="optional package"):
        zoo.build_from_spec(spec, split_data.X_train)


def test_variant_changes_only_the_preprocessor():
    base = zoo.BY_NAME["logistic"]
    v = zoo.variant(
        "logistic", "logistic_meanfill", PreprocessorConfig(numeric_impute="mean"), "mean fill"
    )
    assert v.name == "logistic_meanfill"
    assert v.estimator is base.estimator
    assert v.preprocessor.numeric_impute == "mean"
    # A variant is local: it must not leak into the global registry.
    assert "logistic_meanfill" not in zoo.BY_NAME


def test_preprocessor_rejects_incoherent_config():
    with pytest.raises(ValueError, match="cannot pass NaN"):
        PreprocessorConfig(numeric_impute="none", categorical="onehot")
    with pytest.raises(ValueError, match="unknown numeric_impute"):
        PreprocessorConfig(numeric_impute="knn")


def test_native_nan_config_is_coherent():
    assert NATIVE_NAN.numeric_impute == "none"
    assert NATIVE_NAN.categorical == "ordinal"
