import numpy as np
import pandas as pd
import pytest

from ecri import inference, schema
from ecri.split import LeakageError

CONTRACT = [
    "rank",
    "StuID",
    "p_adequate_growth",
    "risk",
    "risk_percentile",
    "features_present",
    "missing_fields",
    "notes",
]


def test_unseen_students_come_back_most_at_risk_first(bundle, unseen_records):
    result = inference.rank(unseen_records, bundle)
    table = result.table
    assert list(table.columns) == CONTRACT
    assert len(table) == len(unseen_records) == result.n_scored
    assert list(table["rank"]) == list(range(1, len(table) + 1))
    assert table["risk"].is_monotonic_decreasing
    assert np.allclose(table["risk"], 1 - table["p_adequate_growth"])


def test_a_budget_returns_exactly_the_top_k(bundle, unseen_records):
    full = inference.rank(unseen_records, bundle).table
    top = inference.rank(unseen_records, bundle, budget=7).table
    assert len(top) == 7
    pd.testing.assert_frame_equal(top, full.head(7))


def test_a_budget_beyond_the_input_returns_everyone_and_says_so(bundle, unseen_records):
    result = inference.rank(unseen_records.head(5), bundle, budget=50)
    assert len(result.table) == 5
    assert any("exceeds" in w for w in result.warnings)


def test_a_zero_budget_returns_nobody(bundle, unseen_records):
    assert inference.rank(unseen_records, bundle, budget=0).table.empty


@pytest.mark.parametrize("budget,error", [(-1, ValueError), (2.5, TypeError), (True, TypeError)])
def test_bad_budgets_are_refused(bundle, unseen_records, budget, error):
    with pytest.raises(error):
        inference.rank(unseen_records, bundle, budget=budget)


def test_post_baseline_columns_are_refused(bundle, unseen_records):
    with pytest.raises(LeakageError, match="reveal the outcome"):
        inference.rank(unseen_records.assign(**{"RMwidRS_t1.0": 30.0}), bundle)


def test_label_columns_are_refused_even_when_ignoring_unknowns(bundle, unseen_records):
    with pytest.raises(LeakageError):
        inference.rank(unseen_records.assign(RMwidRS_target=1.0), bundle, ignore_unknown=True)


def test_unknown_columns_are_refused_unless_dropped(bundle, unseen_records):
    labelled = unseen_records.assign(TeacherNote="reads well")
    with pytest.raises(inference.InputError, match="does not use"):
        inference.rank(labelled, bundle)
    result = inference.rank(labelled, bundle, ignore_unknown=True)
    assert any("ignored" in w for w in result.warnings)


def test_records_with_missing_fields_are_still_scored(bundle, unseen_records):
    partial = unseen_records.drop(columns=["ORFwc", "grp_rate", "TKPctCrct"])
    partial.loc[partial.index[:5], "SAtoS"] = np.nan
    result = inference.rank(partial, bundle)
    table = result.table.set_index("StuID")
    assert table["p_adequate_growth"].notna().all()
    first = partial["StuID"].iat[0]
    assert table.at[first, "features_present"] == len(bundle.features) - 4
    assert "SAtoS" in table.at[first, "missing_fields"]
    assert any("absent from every record" in w for w in result.warnings)


def test_one_blank_field_does_not_jump_the_queue(bundle, unseen_records):
    """F14 regression: under the T10 pipeline a blank age moved a student half
    the queue. One blank now shifts a student by a few percentile points."""
    base = inference.rank(unseen_records, bundle).table.set_index("StuID")
    shifts = []
    for i in range(0, len(unseen_records), 9):
        blanked = unseen_records.copy()
        blanked.loc[blanked.index[i], "Age1b"] = np.nan
        sid = blanked["StuID"].iat[i]
        moved = inference.rank(blanked, bundle).table.set_index("StuID")
        shifts.append(abs(moved.at[sid, "risk_percentile"] - base.at[sid, "risk_percentile"]))
    assert np.mean(shifts) < 10


def test_text_numbers_are_parsed_and_text_words_are_named(bundle, unseen_records):
    as_text = unseen_records.astype({"RMwidRS": str})
    inference.rank(as_text, bundle)
    broken = as_text.copy()
    broken.loc[broken.index[2], "RMwidRS"] = "twelve"
    with pytest.raises(inference.InputError, match="RMwidRS"):
        inference.rank(broken, bundle)


def test_implausible_values_name_the_student(bundle, unseen_records):
    slip = unseen_records.copy()
    slip.loc[slip.index[0], "Age1b"] = 790.0  # 79.0 with a slipped decimal
    sid = slip["StuID"].iat[0]
    with pytest.raises(inference.InputError, match=f"Age1b=790 for {sid:g}"):
        inference.rank(slip, bundle)


def test_one_mapping_or_a_list_of_mappings_is_accepted(bundle, unseen_records):
    records = unseen_records.head(3).to_dict("records")
    assert len(inference.rank(records[0], bundle).table) == 1
    assert len(inference.rank(records, bundle).table) == 3


def test_rows_are_numbered_when_there_is_no_id(bundle, unseen_records):
    result = inference.rank(unseen_records.drop(columns=["StuID"]), bundle)
    assert result.id_column == "row"
    assert sorted(result.table["row"]) == list(range(1, len(unseen_records) + 1))


def test_equal_scores_keep_input_order(bundle, unseen_records):
    same = pd.concat([unseen_records.head(1)] * 3, ignore_index=True)
    same["StuID"] = [30.0, 10.0, 20.0]
    assert list(inference.rank(same, bundle).table["StuID"]) == [30.0, 10.0, 20.0]


def test_duplicate_ids_are_reported(bundle, unseen_records):
    twice = pd.concat([unseen_records.head(2)] * 2, ignore_index=True)
    assert any("duplicate" in w for w in inference.rank(twice, bundle).warnings)


def test_there_is_no_yes_no_flag_column(bundle, unseen_records):
    # F11: the contract is a ranking. No boolean "at risk" column, by design.
    table = inference.rank(unseen_records, bundle).table
    assert not [c for c in table.columns if table[c].dtype == bool]


def test_notes_mark_scores_resting_on_weak_ground(bundle, unseen_records):
    odd = unseen_records.copy()
    odd.loc[odd.index[0], "Tier"] = 1.0
    odd.loc[odd.index[1], "RMwidRS"] = np.nan
    result = inference.rank(odd, bundle)
    notes = result.table.set_index("StuID")["notes"]
    assert "Tier 1" in notes.at[odd["StuID"].iat[0]]
    assert "no baseline RMwidRS" in notes.at[odd["StuID"].iat[1]]
    assert any("extrapolation" in w for w in result.warnings)


def test_empty_input_is_refused(bundle):
    with pytest.raises(inference.InputError, match="no records"):
        inference.rank(pd.DataFrame(), bundle)


def test_wrong_record_type_is_refused(bundle):
    with pytest.raises(TypeError):
        inference.rank("StuID,RMwidRS", bundle)


def test_risk_percentile_reads_as_share_less_at_risk():
    reference = np.array([0.2, 0.4, 0.6, 0.8])
    p = np.array([0.1, 0.4, 0.9])
    # 0.1: everyone is more likely to grow -> riskier than all; 0.9: than none
    assert np.allclose(inference.risk_percentile(p, reference), [100.0, 62.5, 0.0])


def test_summary_flags_a_synthetic_model(bundle, unseen_records):
    assert "synthetic" in inference.rank(unseen_records, bundle, budget=3).summary()


def test_identifier_columns_pass_through_without_being_features(bundle, unseen_records):
    with_school = unseen_records.assign(**{schema.SCHOOL_ID: 4.0, schema.TEACHER_ID: 9.0})
    assert len(inference.rank(with_school, bundle).table) == len(unseen_records)
