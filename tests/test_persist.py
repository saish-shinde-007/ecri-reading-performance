import json
import shutil

import joblib
import numpy as np
import pytest

from ecri import persist, split


def _copy(bundle_dir, tmp_path):
    target = tmp_path / "copy"
    shutil.copytree(bundle_dir, target)
    return target


def test_round_trip_gives_identical_predictions(bundle, bundle_dir, prepared):
    X, _, _ = split.features_and_target(prepared, "RMwidRS_target")
    loaded = persist.load_bundle(bundle_dir)
    assert np.array_equal(bundle.predict_proba(X), loaded.predict_proba(X))


def test_sidecar_records_what_the_model_card_needs(bundle_dir):
    meta = persist.read_metadata(bundle_dir)
    assert meta["format_version"] == persist.FORMAT_VERSION
    assert meta["target"] == "RMwidRS_target"
    for section, keys in {
        "model": ["name", "description", "preprocessor"],
        "code": ["ecri", "git_sha", "git_dirty"],
        "environment": ["python", "sklearn", "numpy", "pandas"],
        "data": ["source", "rows", "positive_rate", "frame_sha256"],
        "evaluation": ["strategy", "n_folds", "repeats", "seed"],
        "cv": ["auc_mean", "auc_sd", "recall_at_risk_mean", "brier_mean"],
        "calibration": ["method", "in_sample", "brier_raw", "brier_calibrated"],
        "operating_points": ["recommended_cost_ratio", "recommended", "sweep"],
        "population": ["training_tiers", "key_feature"],
    }.items():
        for key in keys:
            assert key in meta[section], f"{section}.{key}"
    assert meta["data"]["source"] == "synthetic"
    assert len(meta["model_sha256"]) == 64


def test_deployed_model_has_no_missingness_flags(bundle):
    # F14: blanks in a teacher's spreadsheet are not the study's missingness.
    assert bundle.metadata["model"]["preprocessor"]["add_indicator"] is False
    names = bundle.pipeline.named_steps["pre"].get_feature_names_out()
    assert not any("missingindicator" in str(n) for n in names)


def test_recorded_cv_auc_reproduces_to_three_places(bundle_dir, prepared):
    loaded = persist.load_bundle(bundle_dir)
    recorded = loaded.metadata["cv"]["auc_mean"]
    assert round(persist.reproduce_cv(loaded, prepared), 3) == round(recorded, 3)


def test_reproduction_refuses_different_data(bundle_dir, prepared):
    loaded = persist.load_bundle(bundle_dir)
    with pytest.raises(persist.IntegrityError, match="not the data"):
        persist.reproduce_cv(loaded, prepared.iloc[:-10])


def test_changed_model_file_is_refused(bundle_dir, tmp_path):
    copy = _copy(bundle_dir, tmp_path)
    with (copy / persist.MODEL_FILE).open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(persist.IntegrityError, match="sha256"):
        persist.load_bundle(copy)


def test_sidecar_feature_list_must_match_the_pipeline(bundle_dir, tmp_path):
    copy = _copy(bundle_dir, tmp_path)
    meta = json.loads((copy / persist.META_FILE).read_text())
    meta["features"] = meta["features"][:-1]
    (copy / persist.META_FILE).write_text(json.dumps(meta))
    with pytest.raises(persist.IntegrityError, match="fitted on"):
        persist.load_bundle(copy)


def test_caller_can_insist_on_a_feature_list(bundle_dir, bundle):
    reordered = list(reversed(bundle.features))
    persist.load_bundle(bundle_dir, expected_features=reordered)  # order is irrelevant
    with pytest.raises(persist.IntegrityError, match="feature list differs"):
        persist.load_bundle(bundle_dir, expected_features=bundle.features[:-1])


def test_prediction_drift_on_the_probe_is_caught(bundle_dir, tmp_path):
    copy = _copy(bundle_dir, tmp_path)
    model_path = copy / persist.MODEL_FILE
    payload = joblib.load(model_path)
    payload["probe_scores"] = payload["probe_scores"] + 1e-6
    joblib.dump(payload, model_path)
    meta = json.loads((copy / persist.META_FILE).read_text())
    meta["model_sha256"] = persist._sha256(model_path)
    (copy / persist.META_FILE).write_text(json.dumps(meta))
    with pytest.raises(persist.IntegrityError, match="probe"):
        persist.load_bundle(copy)


def test_unknown_format_version_is_refused(bundle_dir, tmp_path):
    copy = _copy(bundle_dir, tmp_path)
    meta = json.loads((copy / persist.META_FILE).read_text())
    meta["format_version"] = 99
    (copy / persist.META_FILE).write_text(json.dumps(meta))
    with pytest.raises(persist.IntegrityError, match="format"):
        persist.load_bundle(copy)


def test_missing_bundle_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="ecri model train"):
        persist.load_bundle(tmp_path / "nothing-here")


def test_metadata_is_readable_without_the_model_file(bundle_dir, tmp_path):
    only_json = tmp_path / "meta-only"
    only_json.mkdir()
    shutil.copy(bundle_dir / persist.META_FILE, only_json / persist.META_FILE)
    assert persist.read_metadata(only_json)["target"] == "RMwidRS_target"


def test_reference_scores_are_sorted_probabilities(bundle):
    ref = bundle.reference_scores
    assert np.all(np.diff(ref) >= 0)
    assert ref.min() >= 0 and ref.max() <= 1
    assert len(ref) == bundle.metadata["data"]["rows"]


def test_operating_points_recommend_one_to_one(bundle):
    points = bundle.metadata["operating_points"]
    assert points["recommended_cost_ratio"] == 1.0
    rates = [row["flagged_rate"] for row in points["sweep"]]
    assert rates == sorted(rates)  # a costlier miss can only flag more (F11)


def test_probe_is_synthetic_and_carries_no_identifiers(bundle):
    assert "StuID" not in bundle.probe_X.columns
    assert list(bundle.probe_X.columns) == bundle.features
    assert len(bundle.probe_X) == persist.PROBE_ROWS


def test_default_directory_names_model_and_target():
    assert persist.default_dir("logistic", "RMwidRS_target").name == "logistic-RMwidRS_target"
