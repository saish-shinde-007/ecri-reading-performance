"""Calibration and threshold selection.

An AUC says the ranking is useful. It does not say the number on a screen means
anything, and it does not choose the cut point. A tool that routes children to
reading support needs both: a probability a teacher can read as a probability,
and a threshold justified by what the two mistakes cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402

from . import paths  # noqa: E402
from .metrics import evaluate  # noqa: E402

# Missing a child who will fall short costs more than a second look at a child
# who was going to be fine. 3:1 is a stated assumption, not a measurement —
# change it here and the chosen threshold moves with it.
DEFAULT_COST_RATIO = 3.0


@dataclass(frozen=True)
class CalibrationResult:
    method: str
    brier: float
    # Mean |predicted - observed| over the reliability bins.
    calibration_error: float
    scores: np.ndarray


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    cost_ratio: float
    expected_cost: float
    recall_at_risk: float
    precision_at_risk: float
    flagged_rate: float

    def summary(self) -> str:
        return (
            f"threshold {self.threshold:.2f} at {self.cost_ratio:g}:1 — flags "
            f"{self.flagged_rate:.1%} of students, catches "
            f"{self.recall_at_risk:.1%} of those who fall short, "
            f"{self.precision_at_risk:.1%} of flags are right"
        )


def _calibration_error(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    observed, predicted = calibration_curve(y_true, scores, n_bins=bins, strategy="quantile")
    return float(np.mean(np.abs(observed - predicted)))


def fit_platt(y_true: np.ndarray, scores: np.ndarray) -> LogisticRegression:
    """A one-feature logistic map from raw score to calibrated probability.

    Fit it on OUT-OF-FOLD scores. Fitted on the same model's in-sample scores
    it learns that model's overconfidence on its own training rows instead.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(np.asarray(scores, dtype=float).reshape(-1, 1), np.asarray(y_true, dtype=int))
    return model


def apply_platt(model: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    """Calibrated P(label = 1) for raw scores."""
    column = list(model.classes_).index(1)
    return model.predict_proba(np.asarray(scores, dtype=float).reshape(-1, 1))[:, column]


def fit_calibrators(y_true: np.ndarray, scores: np.ndarray) -> dict[str, CalibrationResult]:
    """Raw, Platt and isotonic, each scored on the same out-of-fold predictions.

    Fitting the calibrator on the same out-of-fold scores it is judged on is
    optimistic by construction, so these numbers are an upper bound. They are
    here to show whether calibration is BROKEN, not to claim a Brier score; a
    nested split would be needed to claim one.
    """
    results = {
        "raw": CalibrationResult(
            "raw",
            float(brier_score_loss(y_true, np.clip(scores, 0, 1))),
            _calibration_error(y_true, scores),
            scores,
        )
    }

    platt_scores = apply_platt(fit_platt(y_true, scores), scores)
    results["platt"] = CalibrationResult(
        "platt",
        float(brier_score_loss(y_true, platt_scores)),
        _calibration_error(y_true, platt_scores),
        platt_scores,
    )

    isotonic = IsotonicRegression(out_of_bounds="clip")
    iso_scores = isotonic.fit_transform(scores, y_true)
    results["isotonic"] = CalibrationResult(
        "isotonic",
        float(brier_score_loss(y_true, iso_scores)),
        _calibration_error(y_true, iso_scores),
        iso_scores,
    )
    return results


def choose_threshold(
    y_true: np.ndarray, scores: np.ndarray, *, cost_ratio: float = DEFAULT_COST_RATIO
) -> ThresholdChoice:
    """Pick the cut point minimising cost_ratio x (missed at-risk) + (false flags).

    "At risk" is label 0 — the student predicted NOT to make adequate growth —
    so a flag is a low score.
    """
    if cost_ratio <= 0:
        raise ValueError("cost_ratio must be positive")
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    best: ThresholdChoice | None = None
    for threshold in np.unique(np.round(scores, 4)):
        flagged = scores < threshold  # predicted to fall short
        at_risk = y_true == 0
        missed = int((at_risk & ~flagged).sum())
        false_flags = int((~at_risk & flagged).sum())
        cost = cost_ratio * missed + false_flags
        caught = int((at_risk & flagged).sum())
        choice = ThresholdChoice(
            threshold=float(threshold),
            cost_ratio=cost_ratio,
            expected_cost=float(cost) / len(y_true),
            recall_at_risk=caught / max(1, int(at_risk.sum())),
            precision_at_risk=caught / max(1, int(flagged.sum())),
            flagged_rate=float(flagged.mean()),
        )
        if best is None or choice.expected_cost < best.expected_cost:
            best = choice
    assert best is not None
    return best


def cost_sweep(
    y_true: np.ndarray, scores: np.ndarray, ratios: tuple[float, ...] = (1, 2, 3, 5, 10)
) -> pd.DataFrame:
    """How the chosen threshold moves as the cost assumption changes."""
    rows = []
    for ratio in ratios:
        choice = choose_threshold(y_true, scores, cost_ratio=ratio)
        rows.append(
            {
                "cost_ratio": ratio,
                "threshold": round(choice.threshold, 3),
                "flagged_rate": round(choice.flagged_rate, 3),
                "recall_at_risk": round(choice.recall_at_risk, 3),
                "precision_at_risk": round(choice.precision_at_risk, 3),
            }
        )
    return pd.DataFrame(rows).set_index("cost_ratio")


def plot_reliability(y_true: np.ndarray, results: dict[str, CalibrationResult], model: str) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=1, label="perfect")
    for name, result in results.items():
        observed, predicted = calibration_curve(
            y_true, result.scores, n_bins=10, strategy="quantile"
        )
        ax.plot(
            predicted,
            observed,
            marker="o",
            markersize=4,
            label=f"{name} (Brier {result.brier:.3f})",
        )
    ax.set_xlabel("predicted P(adequate growth)")
    ax.set_ylabel("observed fraction")
    ax.set_title(f"Reliability — {model}")
    ax.legend(fontsize=8)
    paths.ensure_dirs()
    out = paths.FIGURES / f"calibration_{model}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return paths.rel(out)


def report(
    y_true: np.ndarray, scores: np.ndarray, model: str, *, cost_ratio: float = DEFAULT_COST_RATIO
) -> tuple[dict[str, CalibrationResult], ThresholdChoice, pd.DataFrame, str]:
    """Calibrators, chosen threshold, cost sweep, reliability figure."""
    calibrators = fit_calibrators(y_true, scores)
    choice = choose_threshold(y_true, scores, cost_ratio=cost_ratio)
    sweep = cost_sweep(y_true, scores)
    figure = plot_reliability(y_true, calibrators, model)
    return calibrators, choice, sweep, figure


def metrics_at(y_true: np.ndarray, scores: np.ndarray, threshold: float):
    """Full metric block at a chosen cut point."""
    return evaluate(y_true, scores, threshold=threshold)
