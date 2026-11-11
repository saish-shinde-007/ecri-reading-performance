"""Baselines: the numbers every later model has to beat.

Deliberately dull. Their job is to establish the floor before any tuning, so
that a later 0.63 AUC can be read as progress or as noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ..features import build_preprocessor
from ..metrics import ClassificationMetrics, evaluate, leakage_warning
from ..split import Split

# A shallow tree is here as a non-linear reference point, not a candidate.
BASELINES: dict[str, str] = {
    "majority": "always predicts the majority class",
    "stratified": "random guesses at the training class rate",
    "logistic": "L2 logistic regression on the full feature set",
    "logistic_baseline_only": "logistic regression on the baseline Woodcock score alone",
    "tree_depth3": "decision tree, max_depth=3",
}


@dataclass(frozen=True)
class BaselineResult:
    name: str
    description: str
    metrics: ClassificationMetrics
    warning: str | None = None


def _estimator(name: str, seed: int):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "stratified":
        return DummyClassifier(strategy="stratified", random_state=seed)
    if name in {"logistic", "logistic_baseline_only"}:
        return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    if name == "tree_depth3":
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=seed)
    raise ValueError(f"unknown baseline {name!r}")


def build_baseline(name: str, X: pd.DataFrame, *, seed: int = 20260101) -> Pipeline:
    """Preprocessor + estimator, so every fit sees train-only statistics."""
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; known: {sorted(BASELINES)}")
    columns = ["RMwidRS"] if name == "logistic_baseline_only" else list(X.columns)
    missing = [c for c in columns if c not in X.columns]
    if missing:
        raise ValueError(f"{name} needs columns absent from X: {missing}")
    return Pipeline(
        [
            ("pre", build_preprocessor(X[columns])),
            ("clf", _estimator(name, seed)),
        ]
    )


def _positive_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.classes_)
    if 1 not in classes:  # a fold with one class only
        return np.zeros(len(X))
    return proba[:, classes.index(1)]


def run_baselines(split: Split, *, seed: int = 20260101) -> list[BaselineResult]:
    """Fit every baseline on the training half and score the held-out half."""
    results: list[BaselineResult] = []
    for name, description in BASELINES.items():
        columns = ["RMwidRS"] if name == "logistic_baseline_only" else list(split.X_train.columns)
        if any(c not in split.X_train.columns for c in columns):
            continue
        pipeline = build_baseline(name, split.X_train, seed=seed)
        pipeline.fit(split.X_train[columns], split.y_train)
        scores = _positive_scores(pipeline, split.X_test[columns])
        metrics = evaluate(split.y_test, scores)
        results.append(BaselineResult(name, description, metrics, leakage_warning(metrics)))
    return results


def results_table(results: list[BaselineResult]) -> pd.DataFrame:
    """One row per baseline, sorted by AUC."""
    frame = pd.DataFrame([{"model": r.name, **r.metrics.as_dict()} for r in results]).set_index(
        "model"
    )
    return frame.sort_values("roc_auc", ascending=False)
