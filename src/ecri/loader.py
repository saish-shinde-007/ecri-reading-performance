"""Load the analysis table — real if downloaded, synthetic otherwise."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import datagen, paths, schema


class ContractError(ValueError):
    """A frame does not satisfy the schema in ecri.schema."""


@dataclass(frozen=True)
class Dataset:
    frame: pd.DataFrame
    source: str  # "real" or "synthetic"
    path: Path | None

    @property
    def is_real(self) -> bool:
        return self.source == "real"

    def describe(self) -> str:
        where = self.path.name if self.path else "ecri.datagen"
        return f"{len(self.frame):,} rows x {self.frame.shape[1]} cols from {where} ({self.source})"


def validate(frame: pd.DataFrame, *, require_targets: bool = True) -> None:
    """Fail loudly on anything downstream code assumes."""
    missing = [c for c in schema.ANALYSIS_COLUMNS if c not in frame.columns]
    if require_targets and missing:
        raise ContractError(f"missing contract columns: {missing}")

    if schema.STUDENT_ID not in frame.columns:
        raise ContractError(f"missing identifier {schema.STUDENT_ID}")
    duplicated = int(frame[schema.STUDENT_ID].duplicated().sum())
    if duplicated:
        # F2: the analysis table is one row per student. Several rows per student
        # means the longitudinal file was loaded by mistake.
        raise ContractError(
            f"{duplicated} duplicate {schema.STUDENT_ID} values — "
            "this looks like ECRIRawData.csv, not the analysis table"
        )

    leaked = sorted(c for c in frame.columns if schema.is_leaking(c) and c not in schema.TARGETS)
    if leaked:
        raise ContractError(f"post-baseline columns present, these leak the label: {leaked}")

    for target in schema.TARGETS:
        if target not in frame.columns:
            continue
        values = set(frame[target].dropna().unique())
        if not values <= {0.0, 1.0}:
            raise ContractError(f"{target} must be binary, saw {sorted(values)[:6]}")
        if require_targets and not values:
            raise ContractError(f"{target} has no labelled rows")

    for numeric in schema.NUMERIC_FEATURES:
        if numeric in frame.columns and not pd.api.types.is_numeric_dtype(frame[numeric]):
            raise ContractError(f"{numeric} must be numeric, got {frame[numeric].dtype}")


def load_analysis(
    *, n_synthetic: int = 6803, seed: int = 20260101, allow_synthetic: bool = True
) -> Dataset:
    """Read data/raw/ECRIDataSet.csv, or synthesise a stand-in."""
    if paths.ANALYSIS_CSV.is_file():
        frame = pd.read_csv(paths.ANALYSIS_CSV)
        validate(frame)
        return Dataset(frame=frame, source="real", path=paths.ANALYSIS_CSV)

    if not allow_synthetic:
        raise FileNotFoundError(
            f"{paths.ANALYSIS_CSV} not found and synthetic data was refused; see docs/data.md"
        )
    warnings.warn(
        f"{paths.rel(paths.ANALYSIS_CSV)} not found — using synthetic data. "
        "Numbers are NOT results about real students. See docs/data.md to download.",
        UserWarning,
        stacklevel=2,
    )
    frame = datagen.generate_frame(n_rows=n_synthetic, seed=seed)
    validate(frame)
    return Dataset(frame=frame, source="synthetic", path=None)


def load_school_meta() -> pd.DataFrame | None:
    """School-level context from MetaData.csv, keyed to join on SchlID."""
    if not paths.SCHOOL_META_CSV.is_file():
        return None
    meta = pd.read_csv(paths.SCHOOL_META_CSV)
    return meta.rename(columns={"SchoolID": schema.SCHOOL_ID})


def join_school_meta(
    frame: pd.DataFrame, *, include_composition: bool = False, meta: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Left-join school context onto a student frame. Returns (frame, added).

    Composition columns are opt-in: allocating reading support by a school's
    racial make-up is not a thing to do by default, whatever it does to AUC.
    """
    meta = meta if meta is not None else load_school_meta()
    if meta is None or schema.SCHOOL_ID not in frame.columns:
        return frame, []

    wanted = list(schema.SCHOOL_POVERTY)
    if include_composition:
        wanted += list(schema.SCHOOL_COMPOSITION)
    columns = [c for c in wanted if c in meta.columns]
    if not columns:
        return frame, []

    joined = frame.merge(
        meta[[schema.SCHOOL_ID, *columns]].drop_duplicates(subset=[schema.SCHOOL_ID]),
        on=schema.SCHOOL_ID,
        how="left",
    )
    joined.index = frame.index
    return joined, columns
