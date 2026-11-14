import pytest

from ecri import cli, loader, paths


@pytest.fixture(autouse=True)
def force_synthetic(tmp_path, monkeypatch):
    """CI has no real CSVs; a developer's checkout might. Pin every CLI test to
    synthetic data so the assertions hold either way."""
    monkeypatch.setattr(loader.paths, "ANALYSIS_CSV", tmp_path / "absent.csv")
    monkeypatch.setattr(paths, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(paths, "FIGURES", tmp_path / "reports" / "figures")
    monkeypatch.setattr(paths, "INTERIM", tmp_path / "interim")
    monkeypatch.setattr(paths, "WRITABLE", (paths.INTERIM, paths.REPORTS, paths.FIGURES))


def test_check_passes_on_synthetic_data(capsys):
    assert cli.main(["-q", "check"]) == 0
    out = capsys.readouterr().out
    assert "leakage guard: passed" in out
    assert "synthetic" in out


def test_synth_writes_a_file(capsys):
    assert cli.main(["-q", "synth", "--rows", "400"]) == 0
    assert (paths.INTERIM / "synthetic_ECRIDataSet.csv").is_file()
    assert "400" in capsys.readouterr().out


def test_eda_writes_report_and_figures():
    assert cli.main(["-q", "eda"]) == 0
    assert (paths.REPORTS / "eda.md").is_file()
    assert len(list(paths.FIGURES.glob("*.png"))) == 4


def test_baseline_writes_a_report(capsys):
    assert cli.main(["-q", "baseline"]) == 0
    assert (paths.REPORTS / "baselines.md").is_file()
    assert "logistic" in capsys.readouterr().out


def test_baseline_repeats_report_a_spread(capsys):
    assert cli.main(["-q", "baseline", "--repeats", "3"]) == 0
    out = capsys.readouterr().out
    assert "auc_mean" in out and "auc_sd" in out


def test_target_and_strategy_overrides(capsys):
    assert cli.main(["-q", "--target", "RMwdaRS_target", "--strategy", "student", "check"]) == 0
    assert "leakage guard: passed" in capsys.readouterr().out


def test_unknown_target_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        cli.main(["--target", "SAtoS", "check"])


def test_cv_compares_models_and_writes_a_report(capsys):
    assert (
        cli.main(
            ["-q", "cv", "--models", "logistic,majority", "--repeats", "1", "--bootstrap", "100"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "paired against `logistic`" in out
    assert (paths.REPORTS / "model_comparison.md").is_file()


def test_cv_rejects_unknown_models():
    with pytest.raises(SystemExit):
        cli.main(["-q", "cv", "--models", "transformer"])


def test_impute_reports_a_verdict(capsys):
    assert cli.main(["-q", "impute", "--repeats", "1", "--bootstrap", "100"]) == 0
    assert "imputation choice" in capsys.readouterr().out


def test_importance_writes_a_report(capsys):
    assert cli.main(["-q", "importance", "--models", "logistic", "--repeats", "1"]) == 0
    assert (paths.REPORTS / "importance.md").is_file()
    assert "above noise" in capsys.readouterr().out


def test_calibrate_states_its_cost_assumption(capsys):
    assert cli.main(["-q", "calibrate", "--repeats", "1", "--cost-ratio", "2"]) == 0
    out = capsys.readouterr().out
    assert "2:1" in out
    assert (paths.REPORTS / "calibration.md").is_file()


def test_school_without_metadata_exits_cleanly(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(loader.paths, "SCHOOL_META_CSV", tmp_path / "absent.csv")
    assert cli.main(["-q", "school"]) == 0
    assert "nothing to join" in capsys.readouterr().out
