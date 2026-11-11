"""Train/test splitting, with the leakage guard that makes it mean something."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from . import schema


class LeakageError(AssertionError):
    """A feature matrix contains something that reveals the label."""


@dataclass(frozen=True)
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series | None
    strategy: str

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    def describe(self) -> str:
        return (
            f"{self.strategy}-split: train {self.n_train} rows "
            f"({self.y_train.mean():.1%} positive), test {self.n_test} rows "
            f"({self.y_test.mean():.1%} positive)"
        )


def assert_no_leakage(X: pd.DataFrame, target: str) -> None:
    """Refuse a feature matrix that could reveal the label.

    The label is a t=0 -> t=1 gain (F1), so a post-baseline measurement is a
    perfect predictor and a model built on one is worthless. Identifiers are
    rejected too: SchlID and TeachID are small-cardinality integers that a tree
    will happily memorise into classroom-level label lookup.
    """
    offenders = sorted(c for c in X.columns if schema.is_leaking(c))
    if offenders:
        raise LeakageError(f"feature matrix contains label-revealing columns: {offenders}")
    if target in X.columns:
        raise LeakageError(f"target {target!r} is in the feature matrix")
    ids = sorted(c for c in X.columns if c in schema.IDENTIFIERS)
    if ids:
        raise LeakageError(f"identifiers must not be features: {ids}")


def features_and_target(
    frame: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a prepared frame into (X, y, school groups), guard included."""
    if target not in frame.columns:
        raise ValueError(f"target {target!r} not in the frame")
    roles = schema.roles_for(list(frame.columns))
    X = frame[list(roles.all)].copy()
    y = frame[target].astype(int)
    groups = frame[schema.SCHOOL_ID] if schema.SCHOOL_ID in frame.columns else None
    assert_no_leakage(X, target)
    return X, y, groups


def make_split(
    frame: pd.DataFrame,
    target: str,
    *,
    strategy: str = "school",
    test_size: float = 0.25,
    seed: int = 20260101,
) -> Split:
    """Hold out whole schools ("school") or individual students ("student")."""
    X, y, groups = features_and_target(frame, target)

    if strategy == "school":
        if groups is None:
            raise ValueError(f"school-split needs {schema.SCHOOL_ID} in the frame")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        g_train = groups.iloc[train_idx]
        held_out = set(groups.iloc[test_idx]) & set(g_train)
        if held_out:
            raise LeakageError(f"schools on both sides of the split: {sorted(held_out)}")
    elif strategy == "student":
        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=test_size, random_state=seed, stratify=y
        )
        g_train = groups.iloc[train_idx] if groups is not None else None
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    return Split(
        X_train=X.iloc[train_idx].reset_index(drop=True),
        X_test=X.iloc[test_idx].reset_index(drop=True),
        y_train=y.iloc[train_idx].reset_index(drop=True),
        y_test=y.iloc[test_idx].reset_index(drop=True),
        groups_train=None if g_train is None else g_train.reset_index(drop=True),
        strategy=strategy,
    )


def cv_folds(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
    *,
    strategy: str = "school",
    n_folds: int = 5,
    seed: int = 20260101,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Cross-validation folds matching the chosen split strategy."""
    if strategy == "school":
        if groups is None:
            raise ValueError("school-split CV needs groups")
        n_groups = int(pd.Series(groups).nunique())
        if n_groups < n_folds:
            raise ValueError(f"{n_folds} folds requested but only {n_groups} school(s)")
        yield from StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=seed
        ).split(X, y, groups=groups)
    else:
        yield from StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed
        ).split(X, y)
