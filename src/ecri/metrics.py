"""Metrics for the responder-prediction task."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    n: int
    positive_rate: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    brier: float
    # Recall on the negative class: the students predicted to fall short of
    # adequate growth. Those are the ones an intervention budget should reach,
    # so missing them is the costly error.
    recall_at_risk: float
    tn: int
    fp: int
    fn: int
    tp: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"n={self.n}  AUC={self.roc_auc:.3f}  AP={self.average_precision:.3f}  "
            f"acc={self.accuracy:.3f}  bal-acc={self.balanced_accuracy:.3f}  "
            f"F1={self.f1:.3f}  recall(at-risk)={self.recall_at_risk:.3f}  "
            f"Brier={self.brier:.3f}"
        )


def evaluate(y_true, y_score, *, threshold: float = 0.5) -> ClassificationMetrics:
    """Score probabilistic predictions. y_score is P(adequate growth)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")
    if y_true.size == 0:
        raise ValueError("no rows to evaluate")
    y_pred = (y_score >= threshold).astype(int)

    # A single-class fold makes AUC/AP undefined rather than zero.
    both_classes = len(np.unique(y_true)) == 2
    auc = float(roc_auc_score(y_true, y_score)) if both_classes else float("nan")
    ap = float(average_precision_score(y_true, y_score)) if both_classes else float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return ClassificationMetrics(
        n=int(y_true.size),
        positive_rate=float(y_true.mean()),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=auc,
        average_precision=ap,
        brier=float(brier_score_loss(y_true, np.clip(y_score, 0, 1))),
        recall_at_risk=float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


# Above this, suspect a leak rather than celebrate: the strongest single
# baseline predictor reaches |r| = 0.20 (F3).
SUSPICIOUS_AUC = 0.85


def leakage_warning(metrics: ClassificationMetrics) -> str | None:
    """A note to print when a score is too good to be true."""
    if not np.isnan(metrics.roc_auc) and metrics.roc_auc >= SUSPICIOUS_AUC:
        return (
            f"AUC {metrics.roc_auc:.3f} exceeds the plausible ceiling for this task "
            f"({SUSPICIOUS_AUC}); suspect leakage before reporting it — see docs/findings.md F3"
        )
    return None
