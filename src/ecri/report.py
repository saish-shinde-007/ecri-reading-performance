"""Markdown table rendering.

pandas' own .to_markdown() needs the optional `tabulate` package, which would be
a third-party dependency carried purely to print tables. This is that, in 20
lines, so `pip install -e .` stays minimal and CI needs nothing extra.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:.3g}"
    return str(value)


def to_markdown(frame: pd.DataFrame, *, index_label: str | None = None) -> str:
    """Render a frame as a GitHub-flavoured Markdown table, index included."""
    label = index_label if index_label is not None else (frame.index.name or "")
    headers = [label, *(str(c) for c in frame.columns)]
    rows = [[_cell(idx), *(_cell(v) for v in frame.loc[idx].tolist())] for idx in frame.index]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)) + " |"

    return "\n".join(
        [
            line(headers),
            "|" + "|".join("-" * (w + 2) for w in widths) + "|",
            *(line(r) for r in rows),
        ]
    )
