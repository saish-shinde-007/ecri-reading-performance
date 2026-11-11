"""The ECRI data contract.

Column names and roles are taken from the published `ECRIDataSet.csv` header and
the variable table in the dataset README, not guessed. Definitions quoted in the
comments are the README's own.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── identifiers ──────────────────────────────────────────────────────────────
STUDENT_ID = "StuID"
SCHOOL_ID = "SchlID"
TEACHER_ID = "TeachID"
IDENTIFIERS = (STUDENT_ID, SCHOOL_ID, TEACHER_ID)

# ── targets ──────────────────────────────────────────────────────────────────
# Binary "did the student make adequate growth between t=0 and t=1" labels.
# docs/findings.md derives the exact thresholds (gain >= 27 / >= 11).
TARGET_WORD_ID = "RMwidRS_target"
TARGET_WORD_ATTACK = "RMwdaRS_target"
TARGETS = (TARGET_WORD_ID, TARGET_WORD_ATTACK)

# ── features, grouped by where they come from ────────────────────────────────
STUDENT_CATEGORICAL = (
    "Gender",  # 0/1
    "LEP",     # limited English proficiency
    "SpEd",    # special education status
    "Tier",    # RTI tier 1/2/3
    "Tx",      # 0 = control, 1 = ECRI intervention
    "Cohort",  # study cohort 1/2
)
STUDENT_NUMERIC = ("Age1b",)  # age in months at baseline

# Baseline (t=0) reading assessments. DIBELS, Stanford Achievement, Woodcock.
ASSESSMENT_NUMERIC = (
    "NWFcls",   # Nonsense Word Fluency, correct letter sounds
    "NWFwrc",   # Nonsense Word Fluency, words recoded completely and correctly
    "ORFwc",    # Oral Reading Fluency, words correct per minute
    "SAwrS",    # Stanford Achievement, word reading
    "SAsrS",    # Stanford Achievement, sentence reading
    "SAtoS",    # Stanford Achievement, total reading
    "RMwidRS",  # Woodcock Word Identification, baseline
    "RMwdaRS",  # Woodcock Word Attack, baseline
)

# Classroom and teacher context, constant within a classroom.
CONTEXT_NUMERIC = (
    "Tier2_N",         # number of at-risk readers in the class
    "grp_rate",        # rate of group practice
    "rcmistot",        # Ratings of Classroom Management and Instructional Support
    "gnrl_fid",        # fidelity of implementation
    "TKPctCrct",       # Teacher Knowledge Survey, percent correct
    "Tier2MinRd",      # Tier-2 minutes of reading instruction
    "TYearsTeaching",  # teacher years of experience
)

CATEGORICAL_FEATURES = STUDENT_CATEGORICAL
NUMERIC_FEATURES = STUDENT_NUMERIC + ASSESSMENT_NUMERIC + CONTEXT_NUMERIC
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Single-valued in the analysis table (Keep=1, Time=1.0, t=0.0 throughout), so
# they carry no signal and are dropped rather than fed to a one-hot encoder.
CONSTANT_COLUMNS = ("Keep", "Time", "t")

ANALYSIS_COLUMNS = IDENTIFIERS + CATEGORICAL_FEATURES + STUDENT_NUMERIC + (
    "Tier2_N", "grp_rate", "rcmistot", "gnrl_fid", "TKPctCrct", "Tier2MinRd",
    "TYearsTeaching",
) + CONSTANT_COLUMNS + ASSESSMENT_NUMERIC + TARGETS

# ── leakage deny-list ────────────────────────────────────────────────────────
# The label is a t=0 -> t=1 gain, so any post-baseline measurement reconstructs
# it. These are the names a merge against ECRIRawData.csv would introduce.
_POST_BASELINE_SUFFIXES = ("_t0.5", "_t1", "_t1.0", "_t2", "_t2.0", "_post", "_followup")
_POST_BASELINE_MEASURES = (
    "RMwidRS", "RMwdaRS", "NWFcls", "NWFwrc", "ORFwc",
    "SAwrS", "SAsrS", "SAtoS", "SAwsS", "SArcS",
)
LEAKING_COLUMNS = frozenset(
    [*TARGETS, "Wave", "gain", "growth"]
    + [f"{m}{s}" for m in _POST_BASELINE_MEASURES for s in _POST_BASELINE_SUFFIXES]
)


def is_leaking(column: str) -> bool:
    """True when a column must never appear in a feature matrix."""
    name = column.strip()
    if name in LEAKING_COLUMNS:
        return True
    lowered = name.lower()
    return lowered.endswith("_target") or any(
        lowered.endswith(s) for s in _POST_BASELINE_SUFFIXES
    )


@dataclass(frozen=True)
class ColumnRoles:
    """Which of the contract's columns a given frame actually carries."""

    categorical: tuple[str, ...]
    numeric: tuple[str, ...]

    @property
    def all(self) -> tuple[str, ...]:
        return self.categorical + self.numeric


def roles_for(columns: list[str] | tuple[str, ...]) -> ColumnRoles:
    """Intersect the contract with a frame's columns, keeping contract order."""
    present = set(columns)
    return ColumnRoles(
        categorical=tuple(c for c in CATEGORICAL_FEATURES if c in present),
        numeric=tuple(c for c in NUMERIC_FEATURES if c in present),
    )
