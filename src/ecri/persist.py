"""Saving and loading a trained model, with the provenance needed to trust it.

A bundle is a directory:

    models/<model>-<target>/
        model.joblib   the fitted pipeline, the Platt calibrator, the reference
                       score distribution, and a synthetic probe set
        model.json     the metadata sidecar: what it is, which data it saw,
                       how well it did, and how to reproduce that

The sidecar is what the model card (T36) and the app (T37) read. Its fields are
documented in docs/inference.md and are a contract: add fields freely, but do
not rename or remove one without bumping FORMAT_VERSION.

Loading a joblib file runs code from it, as with any pickle. Load only bundles
you made or trust. The sha256 in the sidecar detects a corrupted or swapped
file; it does not prove who wrote it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from . import __version__, calibrate, datagen, evaluate, paths, schema, split
from .metrics import evaluate as score_metrics
from .models import zoo

FORMAT_VERSION = 1
MODEL_FILE = "model.joblib"
META_FILE = "model.json"

# The probe is synthetic, so the bundle never carries a real student's record.
# Its stored outputs let load() notice when a library upgrade has quietly
# changed what the same model predicts.
PROBE_ROWS = 200
PROBE_SEED = 424242
PROBE_TOLERANCE = 1e-9

# F11: above ~2:1 the cost-optimal threshold flags nearly every student, so the
# operating point marked as recommended is 1:1. The full sweep is stored too,
# for the model card to show how the flag rate moves with the assumption.
RECOMMENDED_COST_RATIO = 1.0
COST_SWEEP = (1.0, 2.0, 3.0, 5.0)

# A tier holding at least this share of training rows counts as "in population".
TIER_SHARE_FLOOR = 0.05


class IntegrityError(RuntimeError):
    """A bundle's files disagree with each other or with this environment."""


@dataclass
class ModelBundle:
    """A trained model plus everything needed to use and audit it."""

    pipeline: Pipeline
    calibrator: Any
    # Calibrated out-of-fold P(adequate growth) for every training student,
    # sorted ascending. inference.rank() places new students against it.
    reference_scores: np.ndarray
    probe_X: pd.DataFrame
    probe_scores: np.ndarray
    metadata: dict[str, Any]

    @property
    def name(self) -> str:
        return self.metadata["model"]["name"]

    @property
    def target(self) -> str:
        return self.metadata["target"]

    @property
    def features(self) -> list[str]:
        return list(self.metadata["features"])

    def raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated P(adequate growth) straight from the pipeline."""
        proba = self.pipeline.predict_proba(X[self.features])
        classes = list(self.pipeline.named_steps["clf"].classes_)
        if 1 not in classes:
            raise IntegrityError("the pipeline never saw a positive label")
        return proba[:, classes.index(1)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated P(adequate growth) for each row of X."""
        return calibrate.apply_platt(self.calibrator, self.raw_scores(X))

    def summary(self) -> str:
        cv = self.metadata["cv"]
        data = self.metadata["data"]
        ev = self.metadata["evaluation"]
        return (
            f"{self.name} → {self.target}\n"
            f"  trained on {data['rows']:,} labelled students ({data['source']} data)\n"
            f"  CV AUC {cv['auc_mean']:.3f} ± {cv['auc_sd']:.3f} "
            f"({ev['strategy']}-split, {ev['n_folds']} folds x {ev['repeats']} repeats)\n"
            f"  {len(self.features)} features; calibrated with Platt on out-of-fold scores"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_sha256(frame: pd.DataFrame, features: list[str], target: str) -> str:
    """Content hash of exactly what the model was fitted and evaluated on.

    Hashes a fixed-format CSV rendering rather than pandas' internal hash, so
    the value does not depend on the pandas version.
    """
    columns = [c for c in features if c in frame.columns] + [target]
    text = frame[columns].to_csv(index=False, float_format="%.10g", lineterminator="\n")
    return hashlib.sha256(text.encode()).hexdigest()


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(paths.ROOT), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()

    try:
        return {
            "git_sha": run("rev-parse", "HEAD"),
            "git_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": None, "git_dirty": None}


def _probe(features: list[str]) -> pd.DataFrame:
    frame = datagen.generate_frame(n_rows=PROBE_ROWS * 4, seed=PROBE_SEED)
    return frame.reindex(columns=features).head(PROBE_ROWS).reset_index(drop=True)


def _population(frame: pd.DataFrame, target: str, features: list[str]) -> dict[str, Any]:
    shares: dict[str, float] = {}
    if "Tier" in frame.columns:
        counts = frame["Tier"].value_counts(normalize=True, dropna=True)
        shares = {f"{float(tier):g}": round(float(share), 4) for tier, share in counts.items()}
    key = schema.BASELINE_OF_TARGET.get(target)
    return {
        "tier_share": shares,
        "training_tiers": sorted(
            float(t) for t, share in shares.items() if share >= TIER_SHARE_FLOOR
        ),
        "key_feature": key if key in features else None,
        "key_feature_missing_rate": (
            round(float(frame[key].isna().mean()), 4) if key in frame.columns else None
        ),
    }


def train_bundle(
    frame: pd.DataFrame,
    target: str,
    model: str = "logistic",
    *,
    strategy: str = "school",
    n_folds: int = 5,
    repeats: int = 5,
    seed: int = 20260101,
    data_info: dict[str, Any] | None = None,
) -> ModelBundle:
    """Evaluate with repeated CV, calibrate on the out-of-fold scores, then fit
    the deployable model on every labelled row.

    This is the scheme sklearn's CalibratedClassifierCV(ensemble=False) uses:
    the calibrator learns from cross-validated predictions, and the final model
    is fitted on all the data.
    """
    X, y, _ = split.features_and_target(frame, target)
    spec = zoo.BY_NAME[model]
    features = zoo.columns_for(spec, X)

    results, _ = evaluate.run_models(
        frame, target, [model], strategy=strategy, n_folds=n_folds, repeats=repeats, seed=seed
    )
    cv = results[model]
    oof = cv.mean_scores
    calibrator = calibrate.fit_platt(cv.y_true, oof)
    calibrated = calibrate.apply_platt(calibrator, oof)

    pipeline = zoo.build(model, X[features], seed=seed)
    pipeline.fit(X[features], y)

    recommended = calibrate.choose_threshold(
        cv.y_true, calibrated, cost_ratio=RECOMMENDED_COST_RATIO
    )
    sweep = calibrate.cost_sweep(cv.y_true, calibrated, ratios=COST_SWEEP)
    at_recommended = score_metrics(cv.y_true, calibrated, threshold=recommended.threshold)

    metadata: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model": {"name": model, "description": spec.description, "family": spec.family},
        "target": target,
        "features": features,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": {"ecri": __version__, **_git_state()},
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "data": {
            "source": "unknown",
            **(data_info or {}),
            "rows": len(frame),
            "positive_rate": round(float(y.mean()), 4),
            "schools": int(frame[schema.SCHOOL_ID].nunique())
            if schema.SCHOOL_ID in frame.columns
            else None,
            "frame_sha256": frame_sha256(frame, features, target),
        },
        "evaluation": {
            "strategy": strategy,
            "n_folds": n_folds,
            "repeats": repeats,
            "seed": seed,
        },
        "cv": {
            "auc_mean": round(cv.mean(), 6),
            "auc_sd": round(cv.sd(), 6),
            "ap_mean": round(cv.mean("average_precision"), 6),
            "balanced_accuracy_mean": round(cv.mean("balanced_accuracy"), 6),
            "recall_at_risk_mean": round(cv.mean("recall_at_risk"), 6),
            "brier_mean": round(cv.mean("brier"), 6),
        },
        "calibration": {
            "method": "platt",
            "fit_on": "out-of-fold scores, averaged over repeats",
            # Scored on the scores it was fitted to, so optimistic (see F11).
            "in_sample": True,
            "brier_raw": round(float(np.mean((oof - cv.y_true) ** 2)), 6),
            "brier_calibrated": round(float(np.mean((calibrated - cv.y_true) ** 2)), 6),
        },
        "operating_points": {
            "note": "Thresholds on calibrated P(adequate growth); a student is "
            "flagged when the score is BELOW the threshold. Rank, don't flag (F11).",
            "recommended_cost_ratio": RECOMMENDED_COST_RATIO,
            "recommended": {
                "threshold": round(recommended.threshold, 6),
                "flagged_rate": round(recommended.flagged_rate, 4),
                "recall_at_risk": round(recommended.recall_at_risk, 4),
                "precision_at_risk": round(recommended.precision_at_risk, 4),
                "balanced_accuracy": round(at_recommended.balanced_accuracy, 4),
            },
            "sweep": [
                {"cost_ratio": float(ratio), **{k: float(v) for k, v in row.items()}}
                for ratio, row in sweep.iterrows()
            ],
        },
        "population": _population(frame, target, features),
    }

    probe_X = _probe(features)
    bundle = ModelBundle(
        pipeline=pipeline,
        calibrator=calibrator,
        reference_scores=np.sort(calibrated),
        probe_X=probe_X,
        probe_scores=np.empty(0),
        metadata=metadata,
    )
    bundle.probe_scores = bundle.predict_proba(probe_X)
    return bundle


def default_dir(model: str, target: str) -> Path:
    return paths.MODELS / f"{model}-{target}"


def save_bundle(bundle: ModelBundle, directory: str | Path | None = None) -> Path:
    """Write model.joblib, then a sidecar that records the file's sha256."""
    directory = Path(directory) if directory else default_dir(bundle.name, bundle.target)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / MODEL_FILE
    joblib.dump(
        {
            "pipeline": bundle.pipeline,
            "calibrator": bundle.calibrator,
            "reference_scores": bundle.reference_scores,
            "probe_X": bundle.probe_X,
            "probe_scores": bundle.probe_scores,
        },
        model_path,
        compress=3,
    )
    bundle.metadata["model_sha256"] = _sha256(model_path)
    (directory / META_FILE).write_text(json.dumps(bundle.metadata, indent=2) + "\n")
    return directory


def read_metadata(directory: str | Path) -> dict[str, Any]:
    """The sidecar alone, without unpickling anything. Safe on any bundle."""
    meta_path = Path(directory) / META_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"no {META_FILE} in {directory} — train one with `ecri model train` or `make model`"
        )
    return json.loads(meta_path.read_text())


def load_bundle(
    directory: str | Path,
    *,
    expected_features: list[str] | None = None,
    verify_probe: bool = True,
) -> ModelBundle:
    """Load a bundle and refuse it if anything does not line up.

    Checks, in order: the format version; the model file's sha256 against the
    sidecar; the pipeline's own feature list against the sidecar's; the
    caller's expected features, when given; and the stored probe predictions,
    which catch a library upgrade that changed what the model outputs.
    """
    directory = Path(directory)
    meta = read_metadata(directory)
    if meta.get("format_version") != FORMAT_VERSION:
        raise IntegrityError(
            f"bundle format {meta.get('format_version')!r}; this code reads {FORMAT_VERSION}"
        )

    model_path = directory / MODEL_FILE
    if not model_path.is_file():
        raise FileNotFoundError(f"{model_path} is missing")
    if _sha256(model_path) != meta.get("model_sha256"):
        raise IntegrityError(
            f"{MODEL_FILE} does not match the sha256 in {META_FILE}: the file was "
            "changed or swapped after it was saved"
        )

    payload = joblib.load(model_path)
    bundle = ModelBundle(metadata=meta, **payload)

    fitted_on = [str(c) for c in getattr(bundle.pipeline, "feature_names_in_", [])]
    if fitted_on != bundle.features:
        raise IntegrityError(
            f"the pipeline was fitted on {fitted_on} but {META_FILE} lists {bundle.features}"
        )

    if expected_features is not None and set(expected_features) != set(bundle.features):
        missing = sorted(set(expected_features) - set(bundle.features))
        extra = sorted(set(bundle.features) - set(expected_features))
        raise IntegrityError(
            f"feature list differs from what the caller expects — "
            f"bundle lacks {missing or 'nothing'}, bundle adds {extra or 'nothing'}"
        )

    if verify_probe:
        now = bundle.predict_proba(bundle.probe_X)
        if not np.allclose(now, bundle.probe_scores, rtol=0.0, atol=PROBE_TOLERANCE):
            drift = float(np.max(np.abs(now - bundle.probe_scores)))
            saved = meta.get("environment", {})
            raise IntegrityError(
                f"predictions on the stored probe moved by up to {drift:.2e}. Saved with "
                f"sklearn {saved.get('sklearn')}, numpy {saved.get('numpy')}; running "
                f"sklearn {sklearn.__version__}, numpy {np.__version__}. Retrain the bundle."
            )
    return bundle


def reproduce_cv(bundle: ModelBundle, frame: pd.DataFrame) -> float:
    """Re-run the recorded evaluation and return the AUC it produces now.

    Refuses a frame that is not byte-for-byte the data the bundle was
    evaluated on, because a different AUC from different data proves nothing.
    """
    expected = bundle.metadata["data"]["frame_sha256"]
    if frame_sha256(frame, bundle.features, bundle.target) != expected:
        raise IntegrityError(
            "this frame is not the data the bundle was evaluated on, so its CV "
            "score cannot be reproduced from it"
        )
    ev = bundle.metadata["evaluation"]
    results, _ = evaluate.run_models(
        frame,
        bundle.target,
        [bundle.name],
        strategy=ev["strategy"],
        n_folds=ev["n_folds"],
        repeats=ev["repeats"],
        seed=ev["seed"],
    )
    return results[bundle.name].mean()
