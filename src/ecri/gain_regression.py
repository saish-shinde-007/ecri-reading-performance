"""Regression on raw Woodcock Word ID gain (T30)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ecri import labels, paths

GAIN_THRESHOLD = 27.0
TARGET_MEASURE = "RMwidRS"
MIN_LEAKAGE_T = 0.5


@dataclass(frozen=True)
class GainComparison:
    """Repeated-CV regression metrics and thresholded-vs-classifier comparison."""

    rmse_mean: float
    rmse_std: float
    mae_mean: float
    mae_std: float
    r2_mean: float
    r2_std: float
    thresholded_roc_auc_mean: float
    thresholded_roc_auc_std: float
    classifier_roc_auc_mean: float
    classifier_roc_auc_std: float
    thresholded_accuracy_mean: float
    classifier_accuracy_mean: float
    paired_auc_diff: float
    ci_low: float
    ci_high: float
    n_students: int
    n_repeats: int
    n_splits: int


def build_gain_target(
    analysis: pd.DataFrame,
    longitudinal: pd.DataFrame,
    *,
    measure: str = TARGET_MEASURE,
    baseline_t: float = labels.BASELINE_T,
    outcome_t: float = labels.OUTCOME_T,
) -> pd.Series:
    """Build t0 -> t1 gain for students in ``analysis`` via pivot on ``t``."""
    wide = longitudinal.pivot_table(index="StuID", columns="t", values=measure, aggfunc="first")
    gain = wide[outcome_t] - wide[baseline_t]
    aligned = analysis["StuID"].map(gain)
    aligned.name = f"{measure}_gain"
    return aligned


def load_gain_target(
    analysis: pd.DataFrame,
    longitudinal_path: Path | None = None,
) -> pd.Series:
    """Load ``ECRIRawData.csv`` and build the raw gain target."""
    csv_path = longitudinal_path or paths.LONGITUDINAL_CSV
    longitudinal = pd.read_csv(csv_path)
    return build_gain_target(analysis, longitudinal)


def assert_no_late_timepoint_leakage(
    features: pd.DataFrame,
    longitudinal: pd.DataFrame,
    *,
    min_t: float = MIN_LEAKAGE_T,
) -> None:
    """Raise ValueError if any feature column matches a t >= 0.5 longitudinal value."""
    late = longitudinal.loc[longitudinal["t"] >= min_t]
    if late.empty:
        return
    for col in features.columns:
        if any(token in col for token in ("_target", "_gain", "_t1", "_t05")):
            raise ValueError(f"Leaky target/late column in features: {col}")


def _make_regressor() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def _make_classifier(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


def evaluate_gain_regression(
    X: pd.DataFrame,
    y_gain: pd.Series,
    *,
    threshold: float = GAIN_THRESHOLD,
    n_splits: int = 5,
    n_repeats: int = 10,
    n_boot: int = 1000,
    seed: int = 42,
) -> GainComparison:
    """Evaluate gain regression and compare thresholded predictions with a classifier."""
    valid = y_gain.notna()
    X_clean = X.loc[valid].reset_index(drop=True)
    y_reg = y_gain.loc[valid].to_numpy(dtype=float)
    y_bin = (y_reg >= threshold).astype(int)

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    n_samples = len(y_reg)
    reg_oof = np.zeros((n_repeats, n_samples), dtype=float)
    clf_oof = np.zeros((n_repeats, n_samples), dtype=float)

    for split_idx, (tr, te) in enumerate(cv.split(X_clean, y_bin)):
        rep = split_idx // n_splits
        reg = _make_regressor()
        reg.fit(X_clean.iloc[tr], y_reg[tr])
        reg_oof[rep, te] = reg.predict(X_clean.iloc[te])

        clf = _make_classifier(seed)
        clf.fit(X_clean.iloc[tr], y_bin[tr])
        clf_oof[rep, te] = clf.predict_proba(X_clean.iloc[te])[:, 1]

    rmses = [root_mean_squared_error(y_reg, reg_oof[r]) for r in range(n_repeats)]
    maes = [mean_absolute_error(y_reg, reg_oof[r]) for r in range(n_repeats)]
    r2s = [r2_score(y_reg, reg_oof[r]) for r in range(n_repeats)]
    reg_aucs = [roc_auc_score(y_bin, reg_oof[r]) for r in range(n_repeats)]
    clf_aucs = [roc_auc_score(y_bin, clf_oof[r]) for r in range(n_repeats)]
    reg_accs = [accuracy_score(y_bin, reg_oof[r] >= threshold) for r in range(n_repeats)]
    clf_accs = [accuracy_score(y_bin, clf_oof[r] >= 0.5) for r in range(n_repeats)]

    reg_mean_score = reg_oof.mean(axis=0)
    clf_mean_score = clf_oof.mean(axis=0)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_samples, size=n_samples)
        if len(np.unique(y_bin[idx])) < 2:
            diffs[b] = roc_auc_score(y_bin, reg_mean_score) - roc_auc_score(y_bin, clf_mean_score)
        else:
            diffs[b] = roc_auc_score(y_bin[idx], reg_mean_score[idx]) - roc_auc_score(
                y_bin[idx], clf_mean_score[idx]
            )
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

    return GainComparison(
        rmse_mean=float(np.mean(rmses)),
        rmse_std=float(np.std(rmses, ddof=1)) if n_repeats > 1 else 0.0,
        mae_mean=float(np.mean(maes)),
        mae_std=float(np.std(maes, ddof=1)) if n_repeats > 1 else 0.0,
        r2_mean=float(np.mean(r2s)),
        r2_std=float(np.std(r2s, ddof=1)) if n_repeats > 1 else 0.0,
        thresholded_roc_auc_mean=float(np.mean(reg_aucs)),
        thresholded_roc_auc_std=float(np.std(reg_aucs, ddof=1)) if n_repeats > 1 else 0.0,
        classifier_roc_auc_mean=float(np.mean(clf_aucs)),
        classifier_roc_auc_std=float(np.std(clf_aucs, ddof=1)) if n_repeats > 1 else 0.0,
        thresholded_accuracy_mean=float(np.mean(reg_accs)),
        classifier_accuracy_mean=float(np.mean(clf_accs)),
        paired_auc_diff=float(np.mean(reg_aucs) - np.mean(clf_aucs)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_students=n_samples,
        n_repeats=n_repeats,
        n_splits=n_splits,
    )
