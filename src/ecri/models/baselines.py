"""Baselines: the numbers every later model has to beat.

Deliberately dull. Their job is to establish the floor before any tuning, so
that a later 0.66 AUC can be read as progress or as noise. The definitions now
live in zoo.py — this module is the single-split runner kept for `ecri
baseline`; anything comparative goes through ecri.evaluate instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..metrics import ClassificationMetrics, evaluate, leakage_warning
from ..split import Split
from . import zoo

BASELINES: dict[str, str] = {
    name: zoo.BY_NAME[name].description for name in zoo.BASELINE_NAMES
}


@dataclass(frozen=True)
class BaselineResult:
    name: str
    description: str
    metrics: ClassificationMetrics
    warning: str | None = None


def build_baseline(name: str, X: pd.DataFrame, *, seed: int = 20260101) -> Pipeline:
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; known: {sorted(BASELINES)}")
    return zoo.build(name, X, seed=seed)


def _positive_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.named_steps["clf"].classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return proba[:, classes.index(1)]


def run_baselines(split: Split, *, seed: int = 20260101) -> list[BaselineResult]:
    """Fit every baseline on the training half and score the held-out half."""
    results: list[BaselineResult] = []
    for name, description in BASELINES.items():
        spec = zoo.BY_NAME[name]
        if spec.columns and any(c not in split.X_train.columns for c in spec.columns):
            continue
        columns = zoo.columns_for(spec, split.X_train)
        pipeline = build_baseline(name, split.X_train[columns], seed=seed)
        pipeline.fit(split.X_train[columns], split.y_train)
        scores = _positive_scores(pipeline, split.X_test[columns])
        metrics = evaluate(split.y_test, scores)
        results.append(BaselineResult(name, description, metrics, leakage_warning(metrics)))
    return results


def results_table(results: list[BaselineResult]) -> pd.DataFrame:
    """One row per baseline, sorted by AUC."""
    frame = pd.DataFrame([{"model": r.name, **r.metrics.as_dict()} for r in results])
    return frame.set_index("model").sort_values("roc_auc", ascending=False)
