import pandas as pd
import pytest

from ecri import loader, schema, split


@pytest.fixture()
def meta(prepared):
    schools = sorted(prepared[schema.SCHOOL_ID].unique())
    rows = []
    for i, school in enumerate(schools):
        rows.append(
            {
                schema.SCHOOL_ID: school,
                "FreeLunch": 0.2 + 0.01 * i,
                **{c: 0.1 for c in schema.SCHOOL_COMPOSITION},
            }
        )
    return pd.DataFrame(rows)


def test_only_poverty_is_joined_by_default(prepared, meta):
    joined, added = loader.join_school_meta(prepared, meta=meta)
    assert added == ["FreeLunch"]
    assert not set(schema.SCHOOL_COMPOSITION) & set(joined.columns)


def test_composition_is_strictly_opt_in(prepared, meta):
    joined, added = loader.join_school_meta(prepared, include_composition=True, meta=meta)
    assert set(schema.SCHOOL_COMPOSITION) <= set(added)


def test_join_preserves_rows_and_order(prepared, meta):
    joined, _ = loader.join_school_meta(prepared, meta=meta)
    assert len(joined) == len(prepared)
    assert joined.index.equals(prepared.index)
    assert (joined[schema.STUDENT_ID].to_numpy() == prepared[schema.STUDENT_ID].to_numpy()).all()


def test_joined_frame_passes_the_leakage_guard(prepared, meta):
    joined, _ = loader.join_school_meta(prepared, include_composition=True, meta=meta)
    X, _, _ = split.features_and_target(joined, "RMwidRS_target")
    assert "FreeLunch" in X.columns


def test_missing_meta_is_a_no_op(prepared, monkeypatch, tmp_path):
    monkeypatch.setattr(loader.paths, "SCHOOL_META_CSV", tmp_path / "absent.csv")
    joined, added = loader.join_school_meta(prepared)
    assert added == [] and joined is prepared
