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
