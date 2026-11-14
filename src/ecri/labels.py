"""Re-deriving the label definition (F1) from the two published files."""

from __future__ import annotations

import numpy as np
import pandas as pd

PAIRS = (("RMwidRS", "RMwidRS_target"), ("RMwdaRS", "RMwdaRS_target"))
BASELINE_T, OUTCOME_T = 0.0, 1.0


def best_gain_threshold(gain: pd.Series, label: pd.Series) -> tuple[float, float]:
    """(threshold, accuracy) of the best single `gain >= threshold` rule."""
    g, y = gain.to_numpy(float), label.to_numpy(float)
    best = (float("nan"), 0.0)
    for thr in np.unique(g):
        acc = float(((g >= thr).astype(float) == y).mean())
        if acc > best[1]:
            best = (float(thr), acc)
    return best


def derive_rules(analysis: pd.DataFrame, longitudinal: pd.DataFrame) -> dict[str, dict]:
    """For each target: the gain threshold that best reproduces it, and how well."""
    rules: dict[str, dict] = {}
    for measure, target in PAIRS:
        labelled = analysis.loc[analysis[target].notna(), ["StuID", target]]
        wide = longitudinal.pivot_table(index="StuID", columns="t", values=measure, aggfunc="first")
        if BASELINE_T not in wide.columns or OUTCOME_T not in wide.columns:
            raise ValueError(
                f"{measure}: t={BASELINE_T:g} and t={OUTCOME_T:g} are not both present"
            )
        merged = labelled.merge(
            (wide[OUTCOME_T] - wide[BASELINE_T]).rename("gain").reset_index(),
            on="StuID",
            how="left",
        ).dropna(subset=["gain", target])
        threshold, accuracy = best_gain_threshold(merged["gain"], merged[target])
        rules[target] = {
            "measure": measure,
            "threshold": threshold,
            "accuracy": accuracy,
            "n": int(len(merged)),
        }
    return rules
