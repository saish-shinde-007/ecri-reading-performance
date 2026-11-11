#!/usr/bin/env python3
"""Reproduce the label definition claimed in docs/findings.md.

Searches for the single gain threshold that best reproduces each *_target column
and reports the accuracy. Needs the real CSVs in data/raw (see docs/data.md);
exits 2 when they are missing so CI can skip rather than fail.

    python tools/verify_target_rule.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ecri import paths

PAIRS = (("RMwidRS", "RMwidRS_target"), ("RMwdaRS", "RMwdaRS_target"))
BASELINE_T, OUTCOME_T = 0.0, 1.0


def best_gain_threshold(gain: pd.Series, label: pd.Series) -> tuple[float, float]:
    """Return (threshold, accuracy) for the best `gain >= threshold` rule."""
    g, y = gain.to_numpy(float), label.to_numpy(float)
    best = (float("nan"), 0.0)
    for thr in np.unique(g):
        acc = float(((g >= thr).astype(float) == y).mean())
        if acc > best[1]:
            best = (float(thr), acc)
    return best


def main() -> int:
    if not (paths.ANALYSIS_CSV.is_file() and paths.LONGITUDINAL_CSV.is_file()):
        print("real ECRI CSVs not in data/raw — see docs/data.md", file=sys.stderr)
        return 2

    analysis = pd.read_csv(paths.ANALYSIS_CSV)
    longitudinal = pd.read_csv(paths.LONGITUDINAL_CSV)

    failures = 0
    for measure, target in PAIRS:
        labelled = analysis.loc[analysis[target].notna(), ["StuID", target]]
        wide = longitudinal.pivot_table(index="StuID", columns="t", values=measure, aggfunc="first")
        if BASELINE_T not in wide.columns or OUTCOME_T not in wide.columns:
            print(f"{measure}: t={BASELINE_T}/{OUTCOME_T} not both present", file=sys.stderr)
            failures += 1
            continue

        merged = labelled.merge(
            (wide[OUTCOME_T] - wide[BASELINE_T]).rename("gain").reset_index(),
            on="StuID",
            how="left",
        ).dropna(subset=["gain", target])

        thr, acc = best_gain_threshold(merged["gain"], merged[target])
        verdict = "EXACT" if acc == 1.0 else f"{acc:.4%}"
        print(
            f"{target}: n={len(merged)}  rule=({measure}@t{OUTCOME_T:g} - "
            f"{measure}@t{BASELINE_T:g}) >= {thr:g}  accuracy={verdict}"
        )
        if acc < 1.0:
            failures += 1

    if failures:
        print(f"\n{failures} rule(s) no longer reproduce exactly.", file=sys.stderr)
        return 1
    print("\nBoth labels reproduce exactly from baseline-to-outcome gain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
