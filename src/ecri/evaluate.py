"""Repeated cross-validation and paired model comparison.

F7 measured a per-split AUC SD near 0.04 on ~200 held-out rows. At that noise
level a single split cannot rank two models, so nothing in phase 2 reports a
single number: every result here is a mean over repeats, and every comparison
is paired on identical folds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import schema, split
from .metrics import ClassificationMetrics, evaluate, leakage_warning
from .models import zoo

Fold = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class FoldPlan:
    """Fold assignments shared by every model, so comparisons are paired."""

    repeats: list[list[Fold]]
    strategy: str
    n_rows: int

    @property
    def n_repeats(self) -> int:
        return len(self.repeats)

    @property
    def n_folds(self) -> int:
        return len(self.repeats[0]) if self.repeats else 0


def make_fold_plan(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    *,
    strategy: str = "school",
    n_folds: int = 5,
    repeats: int = 5,
    seed: int = 20260101,
) -> FoldPlan:
    """Build `repeats` independent fold assignments once, up front."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    plans = [
        list(split.cv_folds(X, y, groups, strategy=strategy, n_folds=n_folds, seed=seed + i))
        for i in range(repeats)
    ]
    return FoldPlan(repeats=plans, strategy=strategy, n_rows=len(X))


@dataclass(frozen=True)
class CVResult:
    """Out-of-fold scores for one model over a whole fold plan."""

    name: str
    description: str
    # (repeats, n_rows): every row is predicted exactly once per repeat.
    oof_scores: np.ndarray
    y_true: np.ndarray
    per_repeat: list[ClassificationMetrics]
    per_fold: list[ClassificationMetrics]

    @property
    def auc(self) -> np.ndarray:
        return np.array([m.roc_auc for m in self.per_repeat], dtype=float)

    def mean(self, metric: str = "roc_auc") -> float:
        return float(np.nanmean([getattr(m, metric) for m in self.per_repeat]))

    def sd(self, metric: str = "roc_auc") -> float:
        values = [getattr(m, metric) for m in self.per_repeat]
        return float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0

    @property
    def mean_scores(self) -> np.ndarray:
        """Per-student score averaged over repeats, for paired tests."""
        return np.nanmean(self.oof_scores, axis=0)

    def summary(self) -> str:
        return (
            f"{self.name:24s} AUC {self.mean():.3f} ± {self.sd():.3f}  "
            f"AP {self.mean('average_precision'):.3f}  "
            f"recall(at-risk) {self.mean('recall_at_risk'):.3f}"
        )


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    plan: FoldPlan,
    model: str | zoo.ModelSpec,
    *,
    seed: int = 20260101,
) -> CVResult:
    """Fit a model on every fold of every repeat and keep out-of-fold scores."""
    spec = zoo.BY_NAME[model] if isinstance(model, str) else model
    name = spec.name
    columns = zoo.columns_for(spec, X)
    y_true = y.to_numpy(int)
    oof = np.full((plan.n_repeats, len(X)), np.nan)
    per_fold: list[ClassificationMetrics] = []

    for r, folds in enumerate(plan.repeats):
        for f, (train_idx, test_idx) in enumerate(folds):
            pipeline = zoo.build_from_spec(
                spec, X.iloc[train_idx][columns], seed=seed + 100 * r + f
            )
            pipeline.fit(X.iloc[train_idx][columns], y.iloc[train_idx])
            scores = _positive_scores(pipeline, X.iloc[test_idx][columns])
            oof[r, test_idx] = scores
            per_fold.append(evaluate(y_true[test_idx], scores))

    per_repeat = [evaluate(y_true, oof[r]) for r in range(plan.n_repeats)]
    return CVResult(
        name=name,
        description=spec.description,
        oof_scores=oof,
        y_true=y_true,
        per_repeat=per_repeat,
        per_fold=per_fold,
    )


def _positive_scores(pipeline, X: pd.DataFrame) -> np.ndarray:
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.named_steps["clf"].classes_)
    if 1 not in classes:  # a degenerate single-class training fold
        return np.zeros(len(X))
    return proba[:, classes.index(1)]


def run_models(
    frame: pd.DataFrame,
    target: str,
    names: list[str],
    *,
    strategy: str = "school",
    n_folds: int = 5,
    repeats: int = 5,
    seed: int = 20260101,
) -> tuple[dict[str, CVResult], FoldPlan]:
    """Cross-validate several models on one shared fold plan."""
    X, y, groups = split.features_and_target(frame, target)
    plan = make_fold_plan(
        X, y, groups, strategy=strategy, n_folds=n_folds, repeats=repeats, seed=seed
    )
    results = {name: cross_validate(X, y, plan, name, seed=seed) for name in names}
    return results, plan


# Imputation strategies compared in T22. Indicator columns are held constant
# except where the variant is explicitly about them.
IMPUTATION_VARIANTS: dict[str, tuple[dict, str]] = {
    "median+indicator": (
        {"numeric_impute": "median", "add_indicator": True},
        "median fill, missingness kept as its own column",
    ),
    "median_only": (
        {"numeric_impute": "median", "add_indicator": False},
        "median fill, missingness discarded",
    ),
    "mean+indicator": (
        {"numeric_impute": "mean", "add_indicator": True},
        "mean fill, missingness kept",
    ),
    "iterative+indicator": (
        {"numeric_impute": "iterative", "add_indicator": True},
        "model-based fill (IterativeImputer), missingness kept",
    ),
}


def compare_imputation(
    frame: pd.DataFrame,
    target: str,
    *,
    base: str = "logistic",
    strategy: str = "school",
    n_folds: int = 5,
    repeats: int = 5,
    seed: int = 20260101,
) -> tuple[dict[str, CVResult], FoldPlan]:
    """Run one estimator under several imputation strategies on shared folds."""
    from .features import NATIVE_NAN, PreprocessorConfig

    X, y, groups = split.features_and_target(frame, target)
    plan = make_fold_plan(
        X, y, groups, strategy=strategy, n_folds=n_folds, repeats=repeats, seed=seed
    )
    specs = [
        zoo.variant(base, name, PreprocessorConfig(**kwargs), description)
        for name, (kwargs, description) in IMPUTATION_VARIANTS.items()
    ]
    # Only a native-NaN estimator can skip imputation entirely.
    if zoo.BY_NAME[base].family == "tree":
        specs.append(
            zoo.variant(base, "native_nan", NATIVE_NAN, "no imputation, NaN passed through")
        )
    results = {spec.name: cross_validate(X, y, plan, spec, seed=seed) for spec in specs}
    return results, plan


def results_table(results: dict[str, CVResult]) -> pd.DataFrame:
    """One row per model, sorted by mean AUC."""
    rows = []
    for name, result in results.items():
        rows.append(
            {
                "model": name,
                "auc_mean": result.mean(),
                "auc_sd": result.sd(),
                "ap_mean": result.mean("average_precision"),
                "balacc_mean": result.mean("balanced_accuracy"),
                "recall_at_risk": result.mean("recall_at_risk"),
                "brier_mean": result.mean("brier"),
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("auc_mean", ascending=False)


@dataclass(frozen=True)
class Comparison:
    """A paired difference between two models on identical folds."""

    better: str
    worse: str
    delta: float
    ci_low: float
    ci_high: float

    @property
    def decisive(self) -> bool:
        """True when the 95% interval excludes zero."""
        return self.ci_low > 0.0

    def summary(self) -> str:
        verdict = "clears noise" if self.decisive else "within noise"
        return (
            f"{self.better} − {self.worse}: ΔAUC {self.delta:+.3f} "
            f"[{self.ci_low:+.3f}, {self.ci_high:+.3f}] — {verdict}"
        )


def paired_bootstrap(
    a: CVResult, b: CVResult, *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> Comparison:
    """Bootstrap the AUC difference over students, holding the folds fixed.

    Resampling students rather than repeats is what makes the interval about
    the data rather than about the fold draw; both models score the same
    students on the same folds, so the difference is paired.
    """
    if not np.array_equal(a.y_true, b.y_true):
        raise ValueError("models were scored on different rows")
    y = a.y_true
    sa, sb = a.mean_scores, b.mean_scores
    observed = float(roc_auc_score(y, sa) - roc_auc_score(y, sb))

    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    drawn = 0
    while drawn < n_boot:
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:  # a resample with one class has no AUC
            continue
        deltas[drawn] = roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx])
        drawn += 1

    low, high = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    better, worse = (a.name, b.name) if observed >= 0 else (b.name, a.name)
    if observed < 0:
        observed, low, high = -observed, -high, -low
    return Comparison(better, worse, observed, float(low), float(high))


def compare_to_reference(
    results: dict[str, CVResult], reference: str = "logistic", **kwargs
) -> list[Comparison]:
    """Every model against the reference, strongest difference first."""
    if reference not in results:
        raise ValueError(f"reference {reference!r} was not cross-validated")
    out = [
        paired_bootstrap(results[name], results[reference], **kwargs)
        for name in results
        if name != reference
    ]
    return sorted(out, key=lambda c: -c.delta)


def warnings_for(results: dict[str, CVResult]) -> list[str]:
    """Leakage notes for any model scoring implausibly well."""
    notes = []
    for name, result in results.items():
        pooled = result.per_repeat[0]
        note = leakage_warning(pooled)
        if note:
            notes.append(f"[{name}] {note}")
    return notes


def oof_frame(result: CVResult, frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Per-student out-of-fold scores, for calibration and slice analysis."""
    out = pd.DataFrame({"y_true": result.y_true, "score": result.mean_scores}, index=frame.index)
    for column in (schema.SCHOOL_ID, "Tier", "LEP", "SpEd", "Gender", "Tx"):
        if column in frame.columns:
            out[column] = frame[column].to_numpy()
    out.attrs["target"] = target
    return out
