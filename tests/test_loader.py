import pandas as pd
import pytest

from ecri import datagen, loader, schema


def test_validate_accepts_a_clean_frame(raw_frame):
    loader.validate(raw_frame)


def test_duplicate_students_rejected(raw_frame):
    doubled = pd.concat([raw_frame, raw_frame], ignore_index=True)
    with pytest.raises(loader.ContractError, match="ECRIRawData"):
        loader.validate(doubled)


def test_post_baseline_column_rejected(raw_frame):
    leaky = raw_frame.assign(**{"RMwidRS_t1.0": 1.0})
    with pytest.raises(loader.ContractError, match="leak"):
        loader.validate(leaky)


def test_non_binary_target_rejected(raw_frame):
    bad = raw_frame.copy()
    bad.loc[bad.index[0], "RMwidRS_target"] = 3.0
    with pytest.raises(loader.ContractError, match="must be binary"):
        loader.validate(bad)


def test_missing_contract_column_rejected(raw_frame):
    with pytest.raises(loader.ContractError, match="missing contract columns"):
        loader.validate(raw_frame.drop(columns=["ORFwc"]))


def test_non_numeric_assessment_rejected(raw_frame):
    bad = raw_frame.assign(ORFwc=raw_frame["ORFwc"].astype(str))
    with pytest.raises(loader.ContractError, match="must be numeric"):
        loader.validate(bad)


def test_synthetic_fallback_warns_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(loader.paths, "ANALYSIS_CSV", tmp_path / "absent.csv")
    with pytest.warns(UserWarning, match="synthetic"):
        dataset = loader.load_analysis(n_synthetic=200)
    assert dataset.source == "synthetic"
    assert not dataset.is_real
    assert "synthetic" in dataset.describe()


def test_synthetic_fallback_can_be_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(loader.paths, "ANALYSIS_CSV", tmp_path / "absent.csv")
    with pytest.raises(FileNotFoundError, match="docs/data.md"):
        loader.load_analysis(allow_synthetic=False)


def test_real_file_is_read_and_validated(tmp_path, monkeypatch):
    path = tmp_path / "ECRIDataSet.csv"
    datagen.generate_frame(n_rows=300, seed=3).to_csv(path, index=False)
    monkeypatch.setattr(loader.paths, "ANALYSIS_CSV", path)
    dataset = loader.load_analysis()
    assert dataset.is_real
    assert list(dataset.frame.columns) == list(schema.ANALYSIS_COLUMNS)
