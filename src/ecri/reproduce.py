"""The reproduction gate (T39).

`ecri reproduce` re-derives the project's headline numbers from the pinned data
and compares them with results/expected.json, the committed record. Every
number must agree to within half a unit in the third decimal.

A mismatch means the code, the data or the environment changed what the
project reports. Find out which before re-recording with `--write`: a change
that moves a published number is a finding, not a formatting update.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from . import clean, evaluate, labels, paths, persist, sources

EXPECTED_PATH = paths.ROOT / "results" / "expected.json"
README_PATH = paths.ROOT / "README.md"
FORMAT = 1
# "Agrees to 3 dp": within half a unit in the third decimal.
TOLERANCE = 5e-4

# Counts of students crossing a fixed cut, rather than continuous scores. They
# are bit-exact on the recording CPU architecture, but on another one a tree
# ensemble can resolve a near-tied split differently, move a score by a tree's
# worth (1/500), and flip a student sitting near the cut: on x86-64 the forest's
# at-risk recall moved by three such flips (6.7e-4) while every continuous
# number held. They are gated on the recording architecture and reported
# elsewhere; every conclusion the project draws rests on the continuous ones.
THRESHOLDED = frozenset({"recall_at_risk"})

CANONICAL: dict[str, Any] = {
    "target": "RMwidRS_target",
    "strategy": "school",
    "n_folds": 5,
    "repeats": 10,
    "bootstrap": 4000,
    "seed": 20260101,
    "reference": "logistic",
    "bundle_model": "logistic",
    "bundle_repeats": 5,
}

# The nine models of F8, in the README's order of introduction. A model joins
# the gate by being added here and the record re-written.
CANONICAL_MODELS: tuple[str, ...] = (
    "logistic",
    "random_forest",
    "hist_gradient_boosting",
    "hgb_native_nan",
    "mlp",
    "tree_depth3",
    "logistic_baseline_only",
    "stratified",
    "majority",
)

DISPLAY = {
    "logistic": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Hist gradient boosting",
    "hgb_native_nan": "Hist GB, native NaN",
    "mlp": "MLP (32-16)",
    "tree_depth3": "Decision tree, depth 3",
    "logistic_baseline_only": "Logistic, baseline score only",
    "stratified": "Stratified guess",
    "majority": "Majority class",
}

# The sections compared number by number. Environment and data are reported,
# not compared: they explain a mismatch rather than constitute one.
COMPARED = ("label_rule", "model_comparison", "deployed_model")

README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"


class ReproductionError(RuntimeError):
    """The gate could not run: wrong data, or a changed canonical config."""


def _r(x: float) -> float:
    return round(float(x), 6)


def run(
    config: dict[str, Any] | None = None,
    *,
    models: tuple[str, ...] = CANONICAL_MODELS,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Re-derive every gated number from the pinned data."""
    config = dict(config or CANONICAL)
    status = sources.status(paths.RAW)
    wrong = {name: state for name, state in status.items() if state != "ok"}
    if wrong:
        raise ReproductionError(
            f"the data in data/raw does not match the pinned commit {sources.COMMIT[:7]}: "
            f"{wrong}. Run `make data`."
        )

    analysis = pd.read_csv(paths.ANALYSIS_CSV)
    longitudinal = pd.read_csv(paths.LONGITUDINAL_CSV)
    target = config["target"]
    frame, _ = clean.prepare(analysis, target)

    progress("label rule (F1) ...")
    rules = labels.derive_rules(analysis, longitudinal)

    progress(
        f"model comparison (F8): {len(models)} models, {config['n_folds']} folds x "
        f"{config['repeats']} repeats ..."
    )
    results, _ = evaluate.run_models(
        frame,
        target,
        list(models),
        strategy=config["strategy"],
        n_folds=config["n_folds"],
        repeats=config["repeats"],
        seed=config["seed"],
    )
    reference = config["reference"]
    comparisons = evaluate.compare_to_reference(
        results, reference, n_boot=config["bootstrap"], seed=config["seed"]
    )
    versus = {}
    for c in comparisons:
        other = c.worse if c.better == reference else c.better
        sign = -1.0 if c.better == reference else 1.0
        low, high = (-c.ci_high, -c.ci_low) if sign < 0 else (c.ci_low, c.ci_high)
        versus[other] = {
            "delta": _r(sign * c.delta),
            "ci_low": _r(low),
            "ci_high": _r(high),
            "decisive": bool(c.decisive),
        }

    progress(
        f"deployed model (T27): {config['bundle_model']}, {config['bundle_repeats']} repeats ..."
    )
    bundle = persist.train_bundle(
        frame,
        target,
        config["bundle_model"],
        strategy=config["strategy"],
        n_folds=config["n_folds"],
        repeats=config["bundle_repeats"],
        seed=config["seed"],
        data_info={"source": "real"},
    )

    return {
        "format": FORMAT,
        "config": config,
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "data": {"commit": sources.COMMIT, "sha256": dict(sources.SHA256)},
        "label_rule": {
            t: {"threshold": _r(r["threshold"]), "accuracy": _r(r["accuracy"]), "n": r["n"]}
            for t, r in rules.items()
        },
        "model_comparison": {
            "models": {
                name: {
                    "auc_mean": _r(result.mean()),
                    "auc_sd": _r(result.sd()),
                    "ap_mean": _r(result.mean("average_precision")),
                    "recall_at_risk": _r(result.mean("recall_at_risk")),
                    "brier_mean": _r(result.mean("brier")),
                }
                for name, result in results.items()
            },
            "vs_reference": versus,
        },
        "deployed_model": {
            "auc_mean": _r(bundle.metadata["cv"]["auc_mean"]),
            "auc_sd": _r(bundle.metadata["cv"]["auc_sd"]),
        },
    }


def _flatten(tree: dict[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _flatten(value, path)
        else:
            yield path, value


@dataclass(frozen=True)
class Check:
    key: str
    expected: Any
    actual: Any
    # Agrees within tolerance (or exactly, for non-floats).
    within: bool
    # False for a thresholded count on a different CPU architecture: a miss is
    # reported, and does not fail the run.
    gated: bool = True

    @property
    def ok(self) -> bool:
        return self.within or not self.gated

    def line(self) -> str:
        mark = "ok  " if self.within else ("FAIL" if self.gated else "info")
        if isinstance(self.expected, float) and isinstance(self.actual, float):
            delta = self.actual - self.expected
            return f"  {mark} {self.key}: {self.expected:.6f} → {self.actual:.6f} ({delta:+.1e})"
        return f"  {mark} {self.key}: {self.expected!r} → {self.actual!r}"


def compare(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    tolerance: float = TOLERANCE,
    strict_counts: bool = True,
) -> tuple[list[Check], list[str]]:
    """(checks, unrecorded keys). Floats within `tolerance`; everything else exact.

    With strict_counts False, a THRESHOLDED key that disagrees is reported
    (gated=False) instead of failing.
    """
    if actual.get("config") != expected.get("config"):
        raise ReproductionError(
            "the canonical configuration differs from the recorded one, so the numbers "
            "are not comparable. Re-record deliberately with `ecri reproduce --write`."
        )
    want = {k: v for s in COMPARED for k, v in _flatten({s: expected.get(s, {})})}
    got = {k: v for s in COMPARED for k, v in _flatten({s: actual.get(s, {})})}

    checks = []
    for key, value in want.items():
        found = got.get(key)
        if found is None:
            ok = False
        elif isinstance(value, bool) or isinstance(found, bool):
            ok = value == found
        elif isinstance(value, int | float) and isinstance(found, int | float):
            ok = abs(float(found) - float(value)) <= tolerance
        else:
            ok = value == found
        gated = strict_counts or key.rsplit(".", 1)[-1] not in THRESHOLDED
        checks.append(Check(key, value, found, ok, gated))
    unrecorded = sorted(set(got) - set(want))
    return checks, unrecorded


def same_architecture(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return actual.get("environment", {}).get("machine") == expected.get("environment", {}).get(
        "machine"
    )


def load_expected(path: Path = EXPECTED_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise ReproductionError(f"no record at {paths.rel(path)}; create one with --write")
    return json.loads(path.read_text())


def write_expected(record: dict[str, Any], path: Path = EXPECTED_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


# ── the README results block ────────────────────────────────────────────────


def _signed(x: float) -> str:
    return f"{x:+.3f}".replace("-", "−")


def readme_block(record: dict[str, Any]) -> str:
    """The README's results table, rendered from the record and nothing else."""
    cfg = record["config"]
    comparison = record["model_comparison"]
    models = comparison["models"]
    versus = comparison["vs_reference"]
    reference = cfg["reference"]
    ordered = sorted(models, key=lambda m: -models[m]["auc_mean"])

    lines = [
        README_START,
        f"{len(models)} models, {cfg['n_folds']}-fold {cfg['strategy']}-disjoint CV repeated "
        f"{cfg['repeats']} times, ROC-AUC on out-of-fold predictions, Word Identification "
        "target. Intervals are a paired bootstrap over students with the folds held fixed "
        f"({cfg['bootstrap']:,} resamples).",
        "",
        f"| Model | AUC | vs {DISPLAY.get(reference, reference).lower()}, 95% CI |",
        "|---|---|---|",
    ]
    for name in ordered:
        label = DISPLAY.get(name, name)
        auc = f"{models[name]['auc_mean']:.3f} ± {models[name]['auc_sd']:.3f}"
        if name == reference:
            lines.append(f"| **{label}** | **{auc}** | — |")
            continue
        v = versus[name]
        verdict = (
            "within noise"
            if not v["decisive"]
            else ("**worse**" if v["delta"] < 0 else "**better**")
        )
        lines.append(
            f"| {label} | {auc} | {_signed(v['delta'])} "
            f"[{_signed(v['ci_low'])}, {_signed(v['ci_high'])}] · {verdict} |"
        )
    lines += [
        "",
        f"*Generated from [`results/expected.json`](results/expected.json) by `ecri readme`. "
        f"`make reproduce` re-derives every number from the pinned data and fails if any "
        f"moves by more than {TOLERANCE:g}.*",
        README_END,
    ]
    return "\n".join(lines)


def render_readme(readme: str, record: dict[str, Any]) -> str:
    if README_START not in readme or README_END not in readme:
        raise ReproductionError(f"README has no {README_START} … {README_END} block")
    head, rest = readme.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    return head + readme_block(record) + tail
