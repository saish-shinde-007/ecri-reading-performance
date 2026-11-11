import pytest

from ecri import schema


@pytest.mark.parametrize(
    "column",
    [
        "RMwidRS_target",
        "RMwdaRS_target",
        "RMwidRS_t1.0",
        "ORFwc_t2",
        "SAtoS_post",
        "NWFcls_followup",
        "Wave",
        "gain",
        "anything_target",
    ],
)
def test_leaking_columns_rejected(column):
    assert schema.is_leaking(column)


@pytest.mark.parametrize("column", list(schema.FEATURES) + ["SchlID", "StuID"])
def test_legitimate_columns_allowed(column):
    assert not schema.is_leaking(column)


def test_baseline_woodcock_is_a_feature_not_a_leak():
    # The label is the t0 -> t1 gain, so the t=0 term is a legal predictor
    # while the t=1 term is a perfect leak. The distinction is the whole point.
    assert not schema.is_leaking("RMwidRS")
    assert schema.is_leaking("RMwidRS_t1.0")


def test_targets_are_not_features():
    assert not set(schema.TARGETS) & set(schema.FEATURES)


def test_constants_are_not_features():
    assert not set(schema.CONSTANT_COLUMNS) & set(schema.FEATURES)


def test_roles_for_keeps_contract_order_and_ignores_strangers():
    roles = schema.roles_for(["ORFwc", "Gender", "not_a_real_column", "RMwidRS"])
    assert roles.categorical == ("Gender",)
    assert roles.numeric == ("ORFwc", "RMwidRS")
    assert "not_a_real_column" not in roles.all


def test_analysis_columns_cover_every_feature_and_target():
    assert set(schema.FEATURES) <= set(schema.ANALYSIS_COLUMNS)
    assert set(schema.TARGETS) <= set(schema.ANALYSIS_COLUMNS)
    assert len(schema.ANALYSIS_COLUMNS) == len(set(schema.ANALYSIS_COLUMNS)) == 30
