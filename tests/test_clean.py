import numpy as np
import pytest

from ecri import clean, schema


def test_label_filter_runs_before_the_missingness_gate(raw_frame):
    """F5: ORFwc is ~79% missing over all rows but ~1% on labelled rows. Gating
    first at any sane threshold would delete the project's better features."""
    all_rows_rate = raw_frame["ORFwc"].isna().mean()
    frame, _ = clean.prepare(raw_frame, "RMwidRS_target", max_missing_rate=0.5)
    assert all_rows_rate > 0.5
    assert "ORFwc" in frame.columns


def test_only_labelled_rows_survive(raw_frame):
    frame, report = clean.prepare(raw_frame, "RMwidRS_target")
    assert frame["RMwidRS_target"].notna().all()
    assert report.rows_out == int(raw_frame["RMwidRS_target"].notna().sum())
    assert report.unlabelled_rows_dropped == report.rows_in - report.rows_out


def test_constant_columns_dropped(prepared):
    for column in schema.CONSTANT_COLUMNS:
        assert column not in prepared.columns


def test_sibling_target_dropped(prepared):
    assert "RMwdaRS_target" not in prepared.columns
    assert "RMwidRS_target" in prepared.columns


def test_target_cast_to_int(prepared):
    assert prepared["RMwidRS_target"].dtype.kind in "iu"


def test_high_missing_column_dropped_when_gate_bites(raw_frame):
    frame = raw_frame.copy()
    frame.loc[frame["RMwidRS_target"].notna(), "ORFwc"] = np.nan
    out, report = clean.prepare(frame, "RMwidRS_target", max_missing_rate=0.9)
    assert "ORFwc" in report.high_missing_columns_dropped
    assert "ORFwc" not in out.columns


def test_out_of_range_flagged_not_clipped(raw_frame):
    frame = raw_frame.copy()
    row = frame.index[frame["RMwidRS_target"].notna()][0]
    frame.loc[row, "gnrl_fid"] = 9.9
    out, report = clean.prepare(frame, "RMwidRS_target")
    assert report.out_of_range["gnrl_fid"] == 1
    # Left in place on purpose: clipping would hide the coding error.
    assert out["gnrl_fid"].max() == pytest.approx(9.9)
    assert "outside the plausible range" in report.summary()


def test_duplicate_students_removed(raw_frame):
    import pandas as pd

    doubled = pd.concat([raw_frame, raw_frame], ignore_index=True)
    frame, report = clean.prepare(doubled, "RMwidRS_target")
    assert report.duplicate_students_dropped == len(raw_frame)
    assert frame[schema.STUDENT_ID].is_unique


def test_whitespace_in_column_names_normalised(raw_frame):
    renamed = raw_frame.rename(columns={"ORFwc": " ORFwc "})
    frame, _ = clean.prepare(renamed, "RMwidRS_target")
    assert "ORFwc" in frame.columns


def test_unknown_target_rejected(raw_frame):
    with pytest.raises(ValueError, match="unknown target"):
        clean.prepare(raw_frame, "SAtoS")
