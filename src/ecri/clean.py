"""Cleaning and assembly of the modelling frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import schema

# Plausible ranges for the published measures. Values outside are recorded as
# suspect and left in place; silently clipping real assessment data would hide
# a coding error rather than surface it.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "Age1b": (48, 120),       # months
    "Tier": (1, 3),
    "Tier2_N": (0, 40),
    "grp_rate": (0, 20),
    "rcmistot": (1, 5),       # rating scale
    "gnrl_fid": (0, 1),       # proportion
    "TKPctCrct": (0, 100),    # percent
    "Tier2MinRd": (0, 600),   # minutes
    "TYearsTeaching": (0, 60),
    "NWFcls": (0, 300),
    "NWFwrc": (0, 150),
    "ORFwc": (0, 300),
    "SAwrS": (200, 800), "SAsrS": (200, 800), "SAtoS": (200, 800),
    "RMwidRS": (0, 120), "RMwdaRS": (0, 60),
}


@dataclass
class CleaningReport:
    """What cleaning did, so a run can be explained after the fact."""

    rows_in: int = 0
    rows_out: int = 0
    duplicate_students_dropped: int = 0
    constant_columns_dropped: list[str] = field(default_factory=list)
    high_missing_columns_dropped: dict[str, float] = field(default_factory=dict)
    out_of_range: dict[str, int] = field(default_factory=dict)
    unlabelled_rows_dropped: int = 0

    def summary(self) -> str:
        lines = [f"rows {self.rows_in:,} -> {self.rows_out:,}"]
        if self.unlabelled_rows_dropped:
            lines.append(f"  dropped {self.unlabelled_rows_dropped:,} unlabelled rows")
        if self.duplicate_students_dropped:
            lines.append(f"  dropped {self.duplicate_students_dropped} duplicate students")
        if self.constant_columns_dropped:
            lines.append(f"  dropped constant columns: {', '.join(self.constant_columns_dropped)}")
        for col, rate in self.high_missing_columns_dropped.items():
            lines.append(f"  dropped {col}: {rate:.1%} missing on labelled rows")
        for col, count in self.out_of_range.items():
            lines.append(f"  WARNING {col}: {count} value(s) outside the plausible range")
        return "\n".join(lines)


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names. Case is left alone — the published
    names are mixed-case and the contract matches them exactly."""
    return frame.rename(columns=lambda c: str(c).strip())


def drop_duplicate_students(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    before = len(frame)
    out = frame.drop_duplicates(subset=[schema.STUDENT_ID], keep="first")
    report.duplicate_students_dropped = before - len(out)
    return out


def drop_constant_columns(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    drop = [
        c for c in frame.columns
        if c not in schema.IDENTIFIERS and frame[c].nunique(dropna=True) <= 1
    ]
    report.constant_columns_dropped = drop
    return frame.drop(columns=drop)


def flag_out_of_range(frame: pd.DataFrame, report: CleaningReport) -> None:
    for col, (low, high) in PLAUSIBLE_RANGES.items():
        if col not in frame.columns:
            continue
        values = frame[col].dropna()
        bad = int(((values < low) | (values > high)).sum())
        if bad:
            report.out_of_range[col] = bad


def prepare(
    frame: pd.DataFrame, target: str, *, max_missing_rate: float = 0.95
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean, then reduce to the rows and columns the model can actually use.

    Order matters: the label filter runs BEFORE the missingness gate. Over all
    rows ORFwc is 78.8% missing, but on labelled rows it is 1.1% (F5), so
    gating first would discard the project's better features.
    """
    if target not in schema.TARGETS:
        raise ValueError(f"unknown target {target!r}")

    report = CleaningReport(rows_in=len(frame))
    out = normalise_columns(frame)
    if target not in out.columns:
        raise ValueError(f"target {target!r} not in the frame")

    out = drop_duplicate_students(out, report)

    labelled = out[out[target].notna()].copy()
    report.unlabelled_rows_dropped = len(out) - len(labelled)

    out = drop_constant_columns(labelled, report)
    flag_out_of_range(out, report)

    for col in schema.roles_for(list(out.columns)).all:
        rate = float(out[col].isna().mean())
        if rate > max_missing_rate:
            report.high_missing_columns_dropped[col] = rate
    out = out.drop(columns=list(report.high_missing_columns_dropped))

    # The other target is a sibling label, not a feature.
    other = [t for t in schema.TARGETS if t != target and t in out.columns]
    out = out.drop(columns=other)

    out[target] = out[target].astype(int)
    report.rows_out = len(out)
    return out.reset_index(drop=True), report
