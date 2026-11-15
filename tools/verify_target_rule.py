#!/usr/bin/env python3
"""Reproduce the label definition claimed in docs/findings.md (F1).

Searches for the single gain threshold that best reproduces each *_target column
and reports the accuracy. Needs the real CSVs in data/raw (see docs/data.md);
exits 2 when they are missing so CI can skip rather than fail.

    python tools/verify_target_rule.py
"""

from __future__ import annotations

import sys

import pandas as pd

from ecri import labels, paths


def main() -> int:
    if not (paths.ANALYSIS_CSV.is_file() and paths.LONGITUDINAL_CSV.is_file()):
        print("real ECRI CSVs not in data/raw — see docs/data.md", file=sys.stderr)
        return 2

    analysis = pd.read_csv(paths.ANALYSIS_CSV)
    rules = labels.derive_rules(analysis, pd.read_csv(paths.LONGITUDINAL_CSV))
    failures = 0
    for target, rule in rules.items():
        verdict = "EXACT" if rule["accuracy"] == 1.0 else f"{rule['accuracy']:.4%}"
        print(
            f"{target}: n={rule['n']}  rule=({rule['measure']}@t1 - {rule['measure']}@t0) "
            f">= {rule['threshold']:g}  accuracy={verdict}"
        )
        failures += rule["accuracy"] < 1.0

    if failures:
        print(f"\n{failures} rule(s) no longer reproduce exactly.", file=sys.stderr)
        return 1
    print("\nBoth labels reproduce exactly from baseline-to-outcome gain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
