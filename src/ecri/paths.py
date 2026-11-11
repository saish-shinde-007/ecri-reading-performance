"""Filesystem layout, resolved from the package location, not the cwd."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"
CONFIGS = ROOT / "configs"

# The three published ECRI files; see docs/data.md for how to fetch them.
ANALYSIS_CSV = RAW / "ECRIDataSet.csv"
LONGITUDINAL_CSV = RAW / "ECRIRawData.csv"
SCHOOL_META_CSV = RAW / "MetaData.csv"

WRITABLE = (INTERIM, PROCESSED, REPORTS, FIGURES, MODELS)


def rel(path: Path) -> str:
    """Repo-relative string when possible, absolute otherwise.

    Path.relative_to raises for anything outside ROOT, which under pytest's
    tmp_path is every artefact. Formatting a path must never be the thing that
    fails a run.
    """
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_dirs() -> None:
    """Create the output directories. Safe to call repeatedly."""
    for d in WRITABLE:
        d.mkdir(parents=True, exist_ok=True)


def real_data_present() -> bool:
    """True when the analysis table has been downloaded."""
    return ANALYSIS_CSV.is_file()
