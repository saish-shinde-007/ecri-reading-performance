"""The feature pipeline: impute, encode, scale — fitted on train only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import schema


@dataclass(frozen=True)
class PreprocessorConfig:
    """How to turn the contract columns into a matrix.

    Exposed as a config rather than hard-coded so T23 can compare imputation
    strategies on identical folds instead of eyeballing two pipelines.
    """

    # "median" | "mean" | "iterative" | "none". "none" passes NaN through for
    # estimators that handle it natively (HistGradientBoosting).
    numeric_impute: str = "median"
    # Keep "was this measured" as its own column. F5: the masks are block
    # structured, so missingness is informative.
    add_indicator: bool = True
    scale: bool = True
    # "onehot" | "ordinal". Ordinal keeps NaN for native-NaN estimators.
    categorical: str = "onehot"

    def __post_init__(self) -> None:
        if self.numeric_impute not in {"median", "mean", "iterative", "none"}:
            raise ValueError(f"unknown numeric_impute {self.numeric_impute!r}")
        if self.categorical not in {"onehot", "ordinal"}:
            raise ValueError(f"unknown categorical {self.categorical!r}")
        if self.numeric_impute == "none" and self.categorical == "onehot":
            raise ValueError(
                "numeric_impute='none' needs categorical='ordinal'; a one-hot "
                "encoder cannot pass NaN through"
            )


NATIVE_NAN = PreprocessorConfig(
    numeric_impute="none", add_indicator=False, scale=False, categorical="ordinal"
)


def _imputer(config: PreprocessorConfig):
    if config.numeric_impute == "iterative":
        # Experimental in sklearn; the import has to come after the opt-in.
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        return IterativeImputer(random_state=0, max_iter=10)
    return SimpleImputer(strategy=config.numeric_impute)


def _numeric_branch(config: PreprocessorConfig):
    """Values imputed and scaled; missingness flags kept as plain 0/1.

    The flags must NOT go through the scaler. A field missing in 0.4% of
    training rows standardises to a ~15-sigma input when it is missing, which
    puts it beyond L2's reach: on the real data a blank age moved the logit by
    6.1, more than any reading measure. Left as 0/1, a flag learned from four
    rows gets a small, regularised coefficient (F14).
    """
    values: list[tuple[str, object]] = []
    if config.numeric_impute != "none":
        values.append(("impute", _imputer(config)))
    if config.scale:
        values.append(("scale", StandardScaler()))
    value_pipe = Pipeline(values) if values else Pipeline([("passthrough", "passthrough")])

    if not config.add_indicator:
        return value_pipe
    return FeatureUnion(
        [
            ("values", value_pipe),
            # error_on_new=False: a field never missing in training may still be
            # blank at inference; it is imputed, and simply has no flag column.
            ("missing", MissingIndicator(features="missing-only", error_on_new=False)),
        ],
        verbose_feature_names_out=False,
    )


def _categorical_branch(config: PreprocessorConfig) -> Pipeline:
    if config.categorical == "ordinal":
        return Pipeline(
            [
                (
                    "encode",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=np.nan,
                    ),
                )
            ]
        )
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


def build_preprocessor(
    X: pd.DataFrame, config: PreprocessorConfig | None = None
) -> ColumnTransformer:
    """A preprocessor for exactly the contract columns present in X."""
    config = config or PreprocessorConfig()
    roles = schema.roles_for(list(X.columns))
    unknown = [c for c in X.columns if c not in roles.all]
    if unknown:
        raise ValueError(f"columns outside the contract reached the pipeline: {unknown}")

    branches = []
    if roles.numeric:
        branches.append(("numeric", _numeric_branch(config), list(roles.numeric)))
    if roles.categorical:
        branches.append(("categorical", _categorical_branch(config), list(roles.categorical)))
    if not branches:
        raise ValueError("no contract features present in X")

    # remainder="drop" so a stray column can never ride through unseen.
    return ColumnTransformer(branches, remainder="drop", verbose_feature_names_out=False)


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Output names of a fitted preprocessor, for importance tables."""
    return [str(n) for n in preprocessor.get_feature_names_out()]
