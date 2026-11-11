import numpy as np
import pytest

from ecri import datagen, schema


def test_columns_match_the_contract_exactly(raw_frame):
    assert list(raw_frame.columns) == list(schema.ANALYSIS_COLUMNS)


def test_one_row_per_student(raw_frame):
    # F2: the analysis table is cross-sectional.
    assert raw_frame[schema.STUDENT_ID].is_unique


def test_deterministic_for_a_seed():
    a = datagen.generate_frame(n_rows=300, seed=11)
    b = datagen.generate_frame(n_rows=300, seed=11)
    c = datagen.generate_frame(n_rows=300, seed=12)
    assert a.equals(b)
    assert not a.equals(c)


def test_labels_cover_a_realistic_minority_mostly_tier_two(raw_frame):
    labelled = raw_frame[raw_frame["RMwidRS_target"].notna()]
    assert 0.11 < len(labelled) / len(raw_frame) < 0.17
    assert (labelled["Tier"] == 2.0).mean() > 0.9


def test_classroom_features_are_constant_within_a_classroom(raw_frame):
    # Without this the school-split is testing nothing.
    for column in ("grp_rate", "rcmistot", "TKPctCrct", "TYearsTeaching"):
        per_teacher = raw_frame.groupby(schema.TEACHER_ID)[column].nunique(dropna=True)
        assert per_teacher.max() <= 1, column


def test_students_nest_inside_schools(raw_frame):
    schools_per_teacher = raw_frame.groupby(schema.TEACHER_ID)[schema.SCHOOL_ID].nunique()
    assert schools_per_teacher.max() == 1
    assert raw_frame[schema.SCHOOL_ID].nunique() == datagen.N_SCHOOLS


def test_missingness_blocks_move_together(raw_frame):
    # F5: an unreturned teacher survey blanks both columns at once.
    a = raw_frame["TKPctCrct"].isna()
    b = raw_frame["TYearsTeaching"].isna()
    assert (a == b).mean() > 0.99


def test_targets_are_binary_and_near_balanced(raw_frame):
    for target in schema.TARGETS:
        values = raw_frame[target].dropna()
        assert set(values.unique()) <= {0.0, 1.0}
        assert 0.4 < values.mean() < 0.7


def test_numeric_ranges_respect_the_observed_bounds(raw_frame):
    for column, (_, _, low, high) in datagen.STUDENT_NUMERIC_STATS.items():
        values = raw_frame[column].dropna()
        assert values.min() >= low and values.max() <= high, column


def test_signal_is_weak_like_the_real_data(raw_frame):
    # F3: nothing should correlate strongly, or the synthetic data would make
    # a leak look normal.
    labelled = raw_frame[raw_frame["RMwidRS_target"].notna()]
    y = labelled["RMwidRS_target"].to_numpy(float)
    for column in schema.NUMERIC_FEATURES:
        values = labelled[column]
        mask = values.notna().to_numpy()
        if mask.sum() < 50 or values[mask].std() == 0:
            continue
        r = np.corrcoef(values[mask].to_numpy(float), y[mask])[0, 1]
        assert abs(r) < 0.45, f"{column} correlates at {r:.2f}"


def test_tiny_frames_are_rejected():
    with pytest.raises(ValueError, match="at least 50"):
        datagen.generate_frame(n_rows=10)
