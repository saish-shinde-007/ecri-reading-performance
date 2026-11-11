"""The feature pipeline: impute, encode, scale — fitted on train only."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import schema


def _numeric_branch() -> Pipeline:
    # add_indicator keeps the fact of missingness as its own feature. F5 shows
    # the masks are block-structured (an unreturned teacher survey, a skipped
    # assessment session), so "was this measured" carries information that
    # imputing to the median would throw away.
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )


def _categorical_branch() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False
                ),
            ),
        ]
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """A preprocessor for exactly the contract columns present in X."""
    roles = schema.roles_for(list(X.columns))
    unknown = [c for c in X.columns if c not in roles.all]
    if unknown:
        raise ValueError(f"columns outside the contract reached the pipeline: {unknown}")

    branches = []
    if roles.numeric:
        branches.append(("numeric", _numeric_branch(), list(roles.numeric)))
    if roles.categorical:
        branches.append(("categorical", _categorical_branch(), list(roles.categorical)))
    if not branches:
        raise ValueError("no contract features present in X")

    # remainder="drop" so a stray column can never ride through unseen.
    return ColumnTransformer(branches, remainder="drop", verbose_feature_names_out=False)


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Output names of a fitted preprocessor, for importance tables."""
    return [str(n) for n in preprocessor.get_feature_names_out()]
