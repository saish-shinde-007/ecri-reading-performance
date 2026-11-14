"""Ranking students for reading support: the contract the app and CLI call.

The output is an ordered queue, never a yes/no flag. When missing a struggling
reader costs even twice a false alarm, the cost-optimal threshold flags nearly
every student (F11), so a flag says little. The useful question is "who first,
given we can help k" — which is what `budget` answers.

Inputs are checked strictly. Label and post-baseline columns are refused
outright, because their presence means the outcome is already known or future
data has been merged in. Unknown columns are refused unless the caller opts to
drop them. Values must be numeric and within plausible ranges.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import schema
from .clean import PLAUSIBLE_RANGES
from .persist import ModelBundle
from .split import LeakageError

# Output columns, in order. Treat as a stable contract (docs/inference.md).
OUTPUT_COLUMNS = (
    "rank",
    "<id>",
    "p_adequate_growth",
    "risk",
    "risk_percentile",
    "features_present",
    "missing_fields",
    "notes",
)
ROW_ID = "row"


class InputError(ValueError):
    """Records that do not fit the model's input contract."""


@dataclass(frozen=True)
class Ranking:
    """The ranked queue plus what the caller needs to show alongside it."""

    table: pd.DataFrame
    warnings: tuple[str, ...]
    model: str
    target: str
    id_column: str
    budget: int | None
    n_scored: int
    data_source: str

    def summary(self) -> str:
        shown = len(self.table)
        head = f"ranked {self.n_scored} student(s) with {self.model} → {self.target}"
        if self.budget is not None:
            head += f"; budget {self.budget}, returning the top {shown}"
        if self.data_source != "real":
            head += f"  [model trained on {self.data_source} data]"
        return head


def _as_frame(records: Any) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, Mapping):
        return pd.DataFrame([dict(records)])
    if (
        isinstance(records, Sequence)
        and not isinstance(records, str | bytes)
        and all(isinstance(r, Mapping) for r in records)
    ):
        return pd.DataFrame([dict(r) for r in records])
    raise TypeError("records must be a DataFrame, a mapping, or a sequence of mappings")


def _coerce_numeric(series: pd.Series, column: str) -> pd.Series:
    text = series.astype("string").str.strip()
    blank = series.isna() | text.fillna("").isin(["", "nan", "NaN", "None", "<NA>"])
    converted = pd.to_numeric(series.where(~blank), errors="coerce")
    bad = ~blank & converted.isna()
    if bad.any():
        examples = sorted({str(v) for v in series[bad].head(3)})
        raise InputError(
            f"{column}: {int(bad.sum())} value(s) are not numbers, e.g. {examples}. "
            "Every model input is a numeric code or score; leave a cell empty if unknown."
        )
    return converted.astype(float)


def _check_ranges(X: pd.DataFrame, ids: pd.Series) -> None:
    problems = []
    for column in X.columns:
        bounds = PLAUSIBLE_RANGES.get(column)
        if bounds is None:
            continue
        low, high = bounds
        bad = X[column].notna() & ((X[column] < low) | (X[column] > high))
        for idx in X.index[bad][:3]:
            problems.append(f"{column}={X.at[idx, column]:g} for {ids.at[idx]}")
        if bad.sum() > 3:
            problems.append(f"... and {int(bad.sum()) - 3} more in {column}")
    if problems:
        raise InputError(
            "value(s) outside the plausible range, likely a data-entry or unit error: "
            + "; ".join(problems)
        )


def _validate_budget(budget: Any) -> int:
    if isinstance(budget, bool) or not isinstance(budget, int | np.integer):
        raise TypeError(f"budget must be a whole number of students, got {budget!r}")
    if budget < 0:
        raise ValueError("budget cannot be negative")
    return int(budget)


def risk_percentile(p_growth: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Share of the reference population LESS at risk than each student, 0-100.

    90 means riskier than 90% of the students the model was evaluated on. Ties
    count half, so a student equal to everyone sits at 50.
    """
    ref = np.sort(np.asarray(reference, dtype=float))
    p = np.asarray(p_growth, dtype=float)
    above = len(ref) - np.searchsorted(ref, p, side="right")  # more likely to grow
    ties = np.searchsorted(ref, p, side="right") - np.searchsorted(ref, p, side="left")
    return 100.0 * (above + 0.5 * ties) / len(ref)


def _notes(X: pd.DataFrame, bundle: ModelBundle) -> tuple[pd.Series, list[str]]:
    """Per-row reasons a score deserves less weight, plus aggregate warnings."""
    population = bundle.metadata.get("population", {})
    tiers = population.get("training_tiers") or []
    key = population.get("key_feature")
    row_notes: list[list[str]] = [[] for _ in range(len(X))]
    warnings = []

    if tiers and "Tier" in X.columns:
        outside = (X["Tier"].notna() & ~X["Tier"].isin(tiers)).to_numpy()
        fit_on = ", ".join(f"{t:g}" for t in tiers)
        for i in np.flatnonzero(outside):
            row_notes[i].append(f"Tier {X['Tier'].iat[i]:g} (model fit on Tier {fit_on})")
        if outside.any():
            warnings.append(
                f"{int(outside.sum())} record(s) are outside the tier the model was fit on; "
                "their scores are extrapolations"
            )

    if key and key in X.columns:
        absent = X[key].isna().to_numpy()
        for i in np.flatnonzero(absent):
            row_notes[i].append(f"no baseline {key}")
        if absent.any():
            warnings.append(
                f"{int(absent.sum())} record(s) lack {key}, the input the score leans on "
                "most (F12); those scores rest on much weaker evidence"
            )
    return pd.Series(["; ".join(n) for n in row_notes], index=X.index), warnings


def rank(
    records: Any,
    bundle: ModelBundle,
    *,
    budget: int | None = None,
    id_column: str = schema.STUDENT_ID,
    ignore_unknown: bool = False,
) -> Ranking:
    """Score records and return them as a queue, most at risk first.

    records        a DataFrame, one mapping, or a sequence of mappings
    budget         return only the top `budget` students
    id_column      carried through to the output; rows are numbered from 1
                   under "row" when it is absent
    ignore_unknown drop columns the model does not use instead of refusing
                   them. Label and post-baseline columns are refused regardless.
    """
    frame = _as_frame(records)
    if frame.empty:
        raise InputError("no records to rank")
    frame = frame.rename(columns=lambda c: str(c).strip())

    leaking = sorted(c for c in frame.columns if schema.is_leaking(c))
    if leaking:
        raise LeakageError(
            f"refusing to rank: {leaking} reveal the outcome. Ranking is for students "
            "whose outcome is not yet known — drop label and post-baseline columns first."
        )

    warnings: list[str] = []
    allowed = set(bundle.features) | set(schema.IDENTIFIERS) | {id_column}
    unknown = sorted(c for c in frame.columns if c not in allowed)
    if unknown:
        if not ignore_unknown:
            raise InputError(
                f"columns the model does not use: {unknown}. Remove them, or pass "
                "ignore_unknown=True to drop them."
            )
        warnings.append(f"ignored {len(unknown)} column(s) the model does not use: {unknown}")
        frame = frame.drop(columns=unknown)

    frame = frame.reset_index(drop=True)
    if id_column in frame.columns:
        ids = frame[id_column]
        out_id = id_column
        duplicated = int(ids.duplicated().sum())
        if duplicated:
            warnings.append(f"{duplicated} duplicate {id_column} value(s) in the input")
    else:
        ids = pd.Series(np.arange(1, len(frame) + 1), index=frame.index)
        out_id = ROW_ID

    X = pd.DataFrame(index=frame.index)
    for column in bundle.features:
        X[column] = _coerce_numeric(frame[column], column) if column in frame.columns else np.nan
    absent = [c for c in bundle.features if c not in frame.columns]
    if absent:
        warnings.append(
            f"{len(absent)} input(s) absent from every record and filled with the "
            f"training median: {absent}"
        )
    _check_ranges(X, ids)

    p_growth = bundle.predict_proba(X)
    notes, population_warnings = _notes(X, bundle)
    warnings += population_warnings
    present = X.notna()

    table = pd.DataFrame(
        {
            out_id: ids.to_numpy(),
            "p_adequate_growth": p_growth,
            "risk": 1.0 - p_growth,
            "risk_percentile": risk_percentile(p_growth, bundle.reference_scores),
            "features_present": present.sum(axis=1).to_numpy(),
            "missing_fields": [
                ", ".join(c for c in bundle.features if not present.at[i, c]) for i in X.index
            ],
            "notes": notes.to_numpy(),
        }
    )
    # Stable sort: equal risks keep input order, so the queue is deterministic.
    table = table.iloc[np.argsort(-table["risk"].to_numpy(), kind="stable")]
    table = table.reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))

    k: int | None = None
    if budget is not None:
        k = _validate_budget(budget)
        if k > len(table):
            warnings.append(f"budget {k} exceeds the {len(table)} record(s) supplied")
        table = table.head(k).reset_index(drop=True)

    return Ranking(
        table=table,
        warnings=tuple(warnings),
        model=bundle.name,
        target=bundle.target,
        id_column=out_id,
        budget=k,
        n_scored=len(frame),
        data_source=str(bundle.metadata.get("data", {}).get("source", "unknown")),
    )
