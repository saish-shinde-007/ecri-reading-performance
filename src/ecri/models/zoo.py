"""The model registry: one name per candidate, one place to add the next."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ..features import NATIVE_NAN, PreprocessorConfig, build_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    family: str  # baseline | linear | tree | neural
    estimator: Callable[[int], object]
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    # Columns to keep, or None for every contract feature present.
    columns: tuple[str, ...] | None = None
    # Third-party module this model needs; the spec is hidden when absent.
    requires: str | None = None
    # Accepts raw NaN, so imputation can be skipped entirely. Explicit rather
    # than inferred from `family`: not every tree in the registry is fed NaN.
    native_nan: bool = False

    @property
    def available(self) -> bool:
        return self.requires is None or importlib.util.find_spec(self.requires) is not None


def _xgboost(seed: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
    )


def _lightgbm(seed: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=400,
        num_leaves=15,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=1,
        verbose=-1,
    )


# Every estimator runs single-threaded. With n_jobs=2 the random forest sums its
# trees' votes in thread-finishing order, and float addition is not
# associative: two runs with the same seed differed by 1.1e-16. Harmless to any
# reported number, but it broke "same inputs, same output" (T39 seed audit).
# Hyper-parameters are deliberately conservative: 946 rows and 22 features
# (F3), so capacity is the enemy. Tuning is T28, and it only means anything
# through the repeated-CV harness.
SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "majority",
        "always predicts the majority class",
        "baseline",
        lambda seed: DummyClassifier(strategy="most_frequent"),
    ),
    ModelSpec(
        "stratified",
        "random guesses at the training class rate",
        "baseline",
        lambda seed: DummyClassifier(strategy="stratified", random_state=seed),
    ),
    ModelSpec(
        "logistic_baseline_only",
        "logistic on the baseline Woodcock score alone",
        "baseline",
        lambda seed: LogisticRegression(max_iter=2000, random_state=seed),
        columns=("RMwidRS",),
    ),
    ModelSpec(
        "tree_depth3",
        "decision tree, max_depth=3",
        "baseline",
        lambda seed: DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=seed),
    ),
    ModelSpec(
        "logistic",
        "L2 logistic regression on the full feature set",
        "linear",
        lambda seed: LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
    ),
    ModelSpec(
        "random_forest",
        "random forest, 500 shallow trees",
        "tree",
        lambda seed: RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=10, n_jobs=1, random_state=seed
        ),
    ),
    ModelSpec(
        "hist_gradient_boosting",
        "histogram gradient boosting, imputed input",
        "tree",
        lambda seed: HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        ),
        native_nan=True,
    ),
    ModelSpec(
        "hgb_native_nan",
        "histogram gradient boosting on raw NaN, no imputation",
        "tree",
        lambda seed: HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        ),
        preprocessor=NATIVE_NAN,
        native_nan=True,
    ),
    ModelSpec(
        "mlp",
        "two-layer MLP, 32-16 units",
        "neural",
        lambda seed: MLPClassifier(
            hidden_layer_sizes=(32, 16),
            alpha=1.0,
            max_iter=2000,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=seed,
        ),
    ),
    ModelSpec(
        "xgboost",
        "XGBoost, shallow and regularised",
        "tree",
        _xgboost,
        preprocessor=NATIVE_NAN,
        requires="xgboost",
        native_nan=True,
    ),
    ModelSpec(
        "lightgbm",
        "LightGBM, shallow and regularised",
        "tree",
        _lightgbm,
        preprocessor=NATIVE_NAN,
        requires="lightgbm",
        native_nan=True,
    ),
)

BY_NAME: dict[str, ModelSpec] = {spec.name: spec for spec in SPECS}
BASELINE_NAMES: tuple[str, ...] = tuple(s.name for s in SPECS if s.family == "baseline") + (
    "logistic",
)


def available_models() -> dict[str, ModelSpec]:
    """Specs whose optional dependency is installed."""
    return {name: spec for name, spec in BY_NAME.items() if spec.available}


def columns_for(spec: ModelSpec, X: pd.DataFrame) -> list[str]:
    if spec.columns is None:
        return list(X.columns)
    missing = [c for c in spec.columns if c not in X.columns]
    if missing:
        raise ValueError(f"{spec.name} needs columns absent from X: {missing}")
    return list(spec.columns)


def build_from_spec(spec: ModelSpec, X: pd.DataFrame, *, seed: int = 20260101) -> Pipeline:
    """Preprocessor + estimator, so every fit sees train-only statistics."""
    if not spec.available:
        raise ImportError(f"{spec.name} needs the optional package {spec.requires!r}")
    columns = columns_for(spec, X)
    return Pipeline(
        [
            ("pre", build_preprocessor(X[columns], spec.preprocessor)),
            ("clf", spec.estimator(seed)),
        ]
    )


def build(name: str, X: pd.DataFrame, *, seed: int = 20260101) -> Pipeline:
    """build_from_spec for a registered name."""
    spec = BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"unknown model {name!r}; known: {sorted(BY_NAME)}")
    return build_from_spec(spec, X, seed=seed)


def variant(base: str, name: str, preprocessor: PreprocessorConfig, description: str) -> ModelSpec:
    """A registered model with a different preprocessor, for A/B work.

    Returned rather than registered: a comparison's variants are local to the
    comparison and have no business in the global registry.
    """
    spec = BY_NAME[base]
    return ModelSpec(
        name=name,
        description=description,
        family=spec.family,
        estimator=spec.estimator,
        preprocessor=preprocessor,
        columns=spec.columns,
        requires=spec.requires,
        native_nan=spec.native_nan,
    )
