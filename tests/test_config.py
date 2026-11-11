import pytest

from ecri.config import Config, SplitConfig, load_config


def test_defaults_are_valid():
    config = Config()
    assert config.target == "RMwidRS_target"
    assert config.split.strategy == "school"


def test_shipped_config_loads():
    config = load_config()
    assert config.target in {"RMwidRS_target", "RMwdaRS_target"}


def test_unknown_target_rejected():
    with pytest.raises(ValueError, match="target must be one of"):
        Config(target="SAtoS")


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="unknown split strategy"):
        Config(split=SplitConfig(strategy="random"))


@pytest.mark.parametrize("size", [0.0, 1.0, -0.2, 1.5])
def test_bad_test_size_rejected(size):
    with pytest.raises(ValueError, match="test_size"):
        Config(split=SplitConfig(test_size=size))


def test_unknown_yaml_key_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("target: RMwidRS_target\nlearning_rate: 0.1\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(path)


def test_yaml_roundtrip(tmp_path):
    path = tmp_path / "ok.yaml"
    path.write_text("target: RMwdaRS_target\nseed: 1\nsplit:\n  strategy: student\n")
    config = load_config(path)
    assert config.target == "RMwdaRS_target"
    assert config.split.strategy == "student"
    assert config.seed == 1
