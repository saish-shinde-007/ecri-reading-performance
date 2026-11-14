"""Model-agnostic feature importance, measured on held-out folds."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from . import paths, split  # noqa: E402
from .evaluate import FoldPlan, make_fold_plan  # noqa: E402
from .models import zoo  # noqa: E402


@dataclass(frozen=True)
class ImportanceResult:
    model: str
    table: pd.DataFrame  # index: feature; columns: mean, sd, n_folds
    figure: str | None = None

    def summary(self, top: int = 10) -> str:
        return self.table.head(top).round(4).to_string()


def permutation_table(
    X: pd.DataFrame,
    y: pd.Series,
    plan: FoldPlan,
    name: str,
    *,
    seed: int = 20260101,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Drop in held-out AUC when each raw feature is shuffled.

    Permuting the RAW column rather than a transformed one keeps the answer in
    the units a teacher can read: "ORFwc matters", not "numeric__ORFwc and its
    missingness indicator jointly matter".
    """
    spec = zoo.BY_NAME[name]
    columns = zoo.columns_for(spec, X)
    per_fold: list[pd.Series] = []

    for r, folds in enumerate(plan.repeats):
        for f, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X.iloc[train_idx][columns], X.iloc[test_idx][columns]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            if y_test.nunique() < 2:  # AUC undefined on a single-class fold
                continue
            pipeline = zoo.build_from_spec(spec, X_train, seed=seed + 100 * r + f)
            pipeline.fit(X_train, y_train)
            result = permutation_importance(
                pipeline,
                X_test,
                y_test,
                scoring="roc_auc",
                n_repeats=n_repeats,
                random_state=seed + f,
                n_jobs=1,
            )
            per_fold.append(pd.Series(result.importances_mean, index=columns))

    if not per_fold:
        raise ValueError("no usable folds for permutation importance")
    stacked = pd.concat(per_fold, axis=1)
    table = pd.DataFrame(
        {
            "auc_drop_mean": stacked.mean(axis=1),
            "auc_drop_sd": stacked.std(axis=1, ddof=1).fillna(0.0),
            "n_folds": stacked.notna().sum(axis=1),
        }
    )
    table.index.name = "feature"
    return table.sort_values("auc_drop_mean", ascending=False)


def plot_importance(table: pd.DataFrame, model: str, top: int = 14) -> str:
    rows = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.34 * len(rows) + 1.2)))
    ax.barh(
        rows.index,
        rows["auc_drop_mean"],
        xerr=rows["auc_drop_sd"],
        color=["#b2182b" if v < 0 else "#2166ac" for v in rows["auc_drop_mean"]],
        error_kw={"elinewidth": 0.8, "ecolor": "#555"},
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("drop in held-out ROC-AUC when shuffled")
    ax.set_title(f"Permutation importance — {model} (bars crossing 0 are noise)")
    paths.ensure_dirs()
    out = paths.FIGURES / f"importance_{model}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return paths.rel(out)


def run(
    frame: pd.DataFrame,
    target: str,
    name: str,
    *,
    strategy: str = "school",
    n_folds: int = 5,
    repeats: int = 3,
    seed: int = 20260101,
    n_repeats: int = 10,
) -> ImportanceResult:
    X, y, groups = split.features_and_target(frame, target)
    plan = make_fold_plan(
        X, y, groups, strategy=strategy, n_folds=n_folds, repeats=repeats, seed=seed
    )
    table = permutation_table(X, y, plan, name, seed=seed, n_repeats=n_repeats)
    return ImportanceResult(model=name, table=table, figure=plot_importance(table, name))


def meaningful(table: pd.DataFrame) -> pd.DataFrame:
    """Features whose mean drop exceeds one SD — i.e. not obviously noise."""
    return table[table["auc_drop_mean"] > table["auc_drop_sd"]]


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Spearman correlation between two models' importance rankings."""
    shared = a.index.intersection(b.index)
    if len(shared) < 3:
        return float("nan")
    ra = a.loc[shared, "auc_drop_mean"].rank()
    rb = b.loc[shared, "auc_drop_mean"].rank()
    return float(np.corrcoef(ra, rb)[0, 1])
