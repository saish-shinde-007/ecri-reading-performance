import math

import pandas as pd

from ecri.report import to_markdown


def test_renders_a_github_table():
    frame = pd.DataFrame({"auc": [0.6612, 0.5]}, index=["logistic", "majority"])
    frame.index.name = "model"
    out = to_markdown(frame).splitlines()
    assert out[0].startswith("| model")
    assert set(out[1]) <= {"|", "-"}
    assert "0.661" in out[2]


def test_nan_renders_as_blank():
    frame = pd.DataFrame({"auc": [math.nan]}, index=["x"])
    assert "nan" not in to_markdown(frame)


def test_index_label_override():
    frame = pd.DataFrame({"v": [1]}, index=["a"])
    assert to_markdown(frame, index_label="thing").startswith("| thing")
