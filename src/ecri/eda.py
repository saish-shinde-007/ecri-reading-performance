"""Exploratory analysis: the figures and the numbers behind docs/findings.md."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # no display in CI or over ssh
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import paths, schema  # noqa: E402
from .report import to_markdown  # noqa: E402

FIGSIZE = (9, 5)


@dataclass(frozen=True)
class EdaResult:
    missingness: pd.DataFrame
    target_balance: pd.DataFrame
    associations: pd.DataFrame
    figures: list[str]


def missingness_table(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Null rates over all rows against the labelled rows only (F5)."""
    labelled = frame[frame[target].notna()]
    cols = [c for c in schema.FEATURES if c in frame.columns]
    return pd.DataFrame(
        {
            "null_pct_all": (frame[cols].isna().mean() * 100).round(1),
            "null_pct_labelled": (labelled[cols].isna().mean() * 100).round(1),
        }
    ).sort_values("null_pct_labelled", ascending=False)


def target_balance_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in schema.TARGETS:
        if target not in frame.columns:
            continue
        values = frame[target].dropna()
        rows.append(
            {
                "target": target,
                "labelled_rows": len(values),
                "positive": int(values.sum()),
                "negative": int((values == 0).sum()),
                "positive_rate": round(float(values.mean()), 3),
            }
        )
    return pd.DataFrame(rows).set_index("target")


def association_table(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Point-biserial correlation of each feature with the label."""
    labelled = frame[frame[target].notna()]
    y = labelled[target].to_numpy(float)
    rows = []
    for col in schema.NUMERIC_FEATURES + schema.CATEGORICAL_FEATURES:
        if col not in labelled.columns:
            continue
        mask = labelled[col].notna().to_numpy()
        if mask.sum() < 30:
            continue
        x = labelled.loc[mask, col].to_numpy(float)
        if np.std(x) == 0 or np.std(y[mask]) == 0:
            continue
        rows.append({"feature": col, "n": int(mask.sum()),
                     "r": round(float(np.corrcoef(x, y[mask])[0, 1]), 3)})
    out = pd.DataFrame(rows).set_index("feature")
    return out.reindex(out["r"].abs().sort_values(ascending=False).index)


def _save(fig: plt.Figure, name: str) -> str:
    paths.ensure_dirs()
    out = paths.FIGURES / name
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out.relative_to(paths.ROOT))


def plot_missingness(table: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    y = np.arange(len(table))
    ax.barh(y + 0.2, table["null_pct_all"], height=0.4, label="all rows")
    ax.barh(y - 0.2, table["null_pct_labelled"], height=0.4, label="labelled rows")
    ax.set_yticks(y, table.index)
    ax.invert_yaxis()
    ax.set_xlabel("% missing")
    ax.set_title("Missingness collapses on the labelled rows")
    ax.legend()
    return _save(fig, "missingness.png")


def plot_associations(table: pd.DataFrame, target: str) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    top = table.head(14).iloc[::-1]
    ax.barh(top.index, top["r"], color=["#b2182b" if v < 0 else "#2166ac" for v in top["r"]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("point-biserial r")
    ax.set_title(f"Feature association with {target} (|r| <= 0.2 throughout)")
    return _save(fig, "associations.png")


def plot_score_distributions(frame: pd.DataFrame, target: str) -> str:
    labelled = frame[frame[target].notna()]
    cols = [c for c in ("RMwidRS", "RMwdaRS", "ORFwc", "SAtoS") if c in labelled.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.4))
    for ax, col in zip(np.atleast_1d(axes), cols, strict=False):
        for value, label in ((0.0, "fell short"), (1.0, "adequate growth")):
            sub = labelled.loc[labelled[target] == value, col].dropna()
            ax.hist(sub, bins=20, alpha=0.55, label=label)
        ax.set_title(col)
        ax.set_xlabel("baseline score")
    np.atleast_1d(axes)[0].set_ylabel("students")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(f"Baseline scores barely separate the classes ({target})")
    return _save(fig, "score_distributions.png")


def plot_school_coverage(frame: pd.DataFrame, target: str) -> str:
    labelled = frame[frame[target].notna()]
    counts = labelled.groupby(schema.SCHOOL_ID).size().sort_values()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(np.arange(len(counts)), counts.to_numpy())
    ax.set_xlabel(f"school (n={len(counts)}, sorted)")
    ax.set_ylabel("labelled students")
    ax.set_title("Labelled students per school — why a school-split varies")
    return _save(fig, "school_coverage.png")


def run(frame: pd.DataFrame, target: str, *, source: str = "unknown") -> EdaResult:
    """Produce every table and figure, and write reports/eda.md."""
    missingness = missingness_table(frame, target)
    balance = target_balance_table(frame)
    associations = association_table(frame, target)
    figures = [
        plot_missingness(missingness),
        plot_associations(associations, target),
        plot_score_distributions(frame, target),
        plot_school_coverage(frame, target),
    ]

    paths.ensure_dirs()
    labelled = frame[frame[target].notna()]
    report = [
        "# EDA report",
        "",
        f"Generated by `ecri eda` from **{source}** data. Target: `{target}`.",
        "",
        f"- rows: {len(frame):,}, of which {len(labelled):,} labelled "
        f"({len(labelled) / len(frame):.1%})",
        f"- schools: {frame[schema.SCHOOL_ID].nunique()}, "
        f"teachers: {frame[schema.TEACHER_ID].nunique()}",
    ]
    if "Tier" in labelled.columns and len(labelled):
        report.append(f"- Tier 2 share of labelled rows: {(labelled['Tier'] == 2).mean():.1%}")
    report += [
        "",
        "## Target balance",
        "",
        to_markdown(balance),
        "",
        "## Missingness (%)",
        "",
        to_markdown(missingness),
        "",
        "## Feature association with the label",
        "",
        to_markdown(associations),
        "",
        "## Figures",
        "",
        *[f"- `{f}`" for f in figures],
        "",
    ]
    (paths.REPORTS / "eda.md").write_text("\n".join(report))
    return EdaResult(missingness, balance, associations, figures)
