"""Synthetic data shaped like ECRIDataSet.csv.

The real data is CC BY-NC-SA 4.0 and git-ignored (docs/data.md), so tests and a
first end-to-end run need a stand-in. The constants below are aggregate
statistics measured from the real file — means, SDs, ranges, null rates and
class balance — not copied records.

Fidelity that matters downstream, and why:
  * students nested in teachers nested in schools, with classroom and teacher
    variables constant within a classroom. A school-disjoint split is pointless
    on data without that nesting.
  * labels on ~14% of rows, drawn mostly from Tier 2, like the real subsample.
  * the block missingness of F5: the teacher-survey columns go missing together.
  * a deliberately weak signal, so baselines land in the real 0.55-0.65 AUC band
    (F3) and a leak shows up as an implausible score here too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import schema

N_SCHOOLS = 44
TEACHERS_PER_SCHOOL = 4
LABELLED_FRACTION = 946 / 6803
TIER2_SHARE_OF_LABELLED = 0.984

# (mean, sd, low, high) measured on the real analysis table.
STUDENT_NUMERIC_STATS = {
    "Age1b": (79.47, 3.94, 70, 94),
    "NWFcls": (43.86, 27.39, 0, 177),
    "NWFwrc": (10.79, 10.41, 0, 57),
    "ORFwc": (12.77, 8.61, 0, 83),
    "SAwrS": (471.05, 58.00, 295, 565),
    "SAsrS": (498.44, 64.76, 321, 596),
    "SAtoS": (504.67, 58.89, 347, 635),
    "RMwidRS": (15.96, 7.11, 0, 62),
    "RMwdaRS": (7.50, 4.41, 0, 30),
}
CLASSROOM_STATS = {
    "Tier2_N": (5.84, 2.69, 1, 14),
    "grp_rate": (1.84, 1.25, 0.02, 5.96),
    "rcmistot": (3.00, 0.36, 2.03, 3.96),
    "gnrl_fid": (0.65, 0.26, 0.0, 1.0),
    "Tier2MinRd": (107.75, 28.56, 0, 127),
}
TEACHER_STATS = {
    "TKPctCrct": (78.70, 8.40, 54.55, 96.97),
    "TYearsTeaching": (14.13, 9.80, 0, 37),
}
INTEGER_COLUMNS = ("Tier2_N", "TYearsTeaching", "Tier2MinRd", "NWFcls", "NWFwrc", "ORFwc")

BINARY_RATES = {"Gender": 0.513, "LEP": 0.172, "SpEd": 0.090, "Tx": 0.492, "Cohort": 0.519}
TIER_PROBS = {1.0: 0.527, 2.0: 0.242, 3.0: 0.231}

# Null rates on the labelled rows (F5), which is where they bite.
NULL_RATES_LABELLED = {
    "Tier2MinRd": 0.248,
    "TKPctCrct": 0.223,
    "TYearsTeaching": 0.223,
    "grp_rate": 0.056,
    "rcmistot": 0.035,
    "gnrl_fid": 0.032,
    "NWFcls": 0.011,
    "NWFwrc": 0.011,
    "ORFwc": 0.011,
    "Age1b": 0.004,
    "SpEd": 0.004,
    "LEP": 0.003,
}
# Null rates on the unlabelled rows, which dominate the all-rows figures.
NULL_RATES_UNLABELLED = {
    "Age1b": 1.0,
    "RMwidRS": 1.0,
    "RMwdaRS": 1.0,
    "ORFwc": 0.906,
    "NWFcls": 0.648,
    "NWFwrc": 0.648,
    "grp_rate": 0.479,
    "rcmistot": 0.459,
    "Tier2MinRd": 0.258,
    "TKPctCrct": 0.256,
    "TYearsTeaching": 0.256,
    "Tier": 0.134,
    "gnrl_fid": 0.080,
    "SAwrS": 0.128,
    "SAsrS": 0.128,
    "SAtoS": 0.128,
    "LEP": 0.061,
    "SpEd": 0.061,
    "Gender": 0.002,
}
# Columns missing together, as one survey or one assessment session.
MISSINGNESS_BLOCKS = (
    ("TKPctCrct", "TYearsTeaching"),
    ("NWFcls", "NWFwrc"),
    ("SAwrS", "SAsrS", "SAtoS"),
)

POSITIVE_RATE = {"RMwidRS_target": 0.527, "RMwdaRS_target": 0.576}
# Weights on standardised features for the latent gain. Signs and magnitudes
# follow the real point-biserial correlations in F3 — baseline level is mildly
# NEGATIVE because low starters have more headroom to gain.
GAIN_WEIGHTS = {
    "RMwidRS": -0.42,
    "SAsrS": -0.26,
    "ORFwc": -0.26,
    "SAtoS": -0.14,
    "Tier2_N": 0.15,
    "gnrl_fid": 0.13,
    "SAwrS": -0.09,
    "Tx": 0.10,
}
# Noise SD relative to the signal. Tuned so the achievable AUC lands near 0.62.
GAIN_NOISE = 1.9


@dataclass(frozen=True)
class SyntheticData:
    frame: pd.DataFrame
    labelled_index: pd.Index


def _truncated_normal(
    rng: np.random.Generator, mean: float, sd: float, low: float, high: float, size: int
) -> np.ndarray:
    """Normal draws clipped to the observed range. Clipping is deliberate: it
    reproduces the real ceilings (Tier2MinRd piles up at 127, gnrl_fid at 1.0)."""
    return np.clip(rng.normal(mean, sd, size), low, high)


def _standardise(values: pd.Series) -> np.ndarray:
    v = values.to_numpy(float)
    sd = np.nanstd(v)
    return np.zeros_like(v) if sd == 0 else (v - np.nanmean(v)) / sd


def generate(n_rows: int = 6803, seed: int = 20260101) -> SyntheticData:
    """Build a frame with the same columns and dtypes as the analysis table."""
    if n_rows < 50:
        raise ValueError("n_rows must be at least 50 to keep the nesting meaningful")
    rng = np.random.default_rng(seed)

    # ── nesting: student -> teacher -> school ────────────────────────────────
    n_teachers = N_SCHOOLS * TEACHERS_PER_SCHOOL
    teacher_of_row = rng.integers(0, n_teachers, n_rows)
    school_of_teacher = np.repeat(np.arange(1, N_SCHOOLS + 1), TEACHERS_PER_SCHOOL)

    df = pd.DataFrame(
        {
            schema.STUDENT_ID: np.arange(1, n_rows + 1, dtype=float),
            schema.SCHOOL_ID: school_of_teacher[teacher_of_row].astype(float),
            schema.TEACHER_ID: (teacher_of_row + 3).astype(float),
        }
    )

    # ── student-level draws ─────────────────────────────────────────────────
    for col, rate in BINARY_RATES.items():
        df[col] = (rng.random(n_rows) < rate).astype(float)
    df["Cohort"] = (df["Cohort"] + 1).astype(int)
    df["Tx"] = df["Tx"].astype(int)
    tiers, probs = zip(*TIER_PROBS.items(), strict=True)
    df["Tier"] = rng.choice(tiers, size=n_rows, p=np.asarray(probs) / sum(probs))

    for col, (mean, sd, low, high) in STUDENT_NUMERIC_STATS.items():
        df[col] = _truncated_normal(rng, mean, sd, low, high, n_rows)

    # ── classroom and teacher draws, constant within a classroom ────────────
    for col, (mean, sd, low, high) in {**CLASSROOM_STATS, **TEACHER_STATS}.items():
        per_teacher = _truncated_normal(rng, mean, sd, low, high, n_teachers)
        df[col] = per_teacher[teacher_of_row]

    for col in INTEGER_COLUMNS:
        df[col] = np.round(df[col])

    # Constant in the real file (F2); kept so the schema matches column-for-column.
    df["Keep"] = 1.0
    df["Time"] = 1.0
    df["t"] = 0.0

    # ── which rows carry labels: the Woodcock subsample, mostly Tier 2 ──────
    n_labelled = max(2, int(round(n_rows * LABELLED_FRACTION)))
    tier2 = df.index[df["Tier"] == 2.0].to_numpy()
    other = df.index[df["Tier"] != 2.0].to_numpy()
    n_tier2 = min(len(tier2), int(round(n_labelled * TIER2_SHARE_OF_LABELLED)))
    chosen = np.concatenate(
        [
            rng.choice(tier2, size=n_tier2, replace=False),
            rng.choice(other, size=min(len(other), n_labelled - n_tier2), replace=False),
        ]
    )
    labelled = pd.Index(np.sort(chosen))

    # ── latent gain -> binary label, weak signal by construction ────────────
    signal = np.zeros(n_rows)
    for col, weight in GAIN_WEIGHTS.items():
        signal += weight * _standardise(df[col])
    latent = signal + rng.normal(0, GAIN_NOISE, n_rows)
    for target, positive_rate in POSITIVE_RATE.items():
        column = pd.Series(np.nan, index=df.index, dtype=float)
        sub = latent[labelled] + rng.normal(0, 0.35, len(labelled))
        column.loc[labelled] = (sub >= np.quantile(sub, 1 - positive_rate)).astype(float)
        df[target] = column

    df = _apply_missingness(df, rng, labelled)
    return SyntheticData(frame=df[list(schema.ANALYSIS_COLUMNS)], labelled_index=labelled)


def _apply_missingness(
    df: pd.DataFrame, rng: np.random.Generator, labelled: pd.Index
) -> pd.DataFrame:
    """Blank values at the measured rates, keeping co-missing columns together."""
    blocked = {c for block in MISSINGNESS_BLOCKS for c in block}
    groups = [*MISSINGNESS_BLOCKS, *[(c,) for c in schema.FEATURES if c not in blocked]]

    for rows, rates in (
        (labelled, NULL_RATES_LABELLED),
        (df.index.difference(labelled), NULL_RATES_UNLABELLED),
    ):
        if len(rows) == 0:
            continue
        for block in groups:
            rate = max((rates.get(c, 0.0) for c in block), default=0.0)
            if rate <= 0:
                continue
            # One draw per block, so the whole survey or session goes missing at once.
            mask = rows[rng.random(len(rows)) < rate]
            for col in block:
                if col in df.columns:
                    df.loc[mask, col] = np.nan
    return df


def generate_frame(n_rows: int = 6803, seed: int = 20260101) -> pd.DataFrame:
    """generate() without the index bookkeeping."""
    return generate(n_rows=n_rows, seed=seed).frame
