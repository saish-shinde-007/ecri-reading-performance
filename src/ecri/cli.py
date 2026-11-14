"""Command line entry point: `ecri {synth,eda,baseline,check}`."""

from __future__ import annotations

import argparse
import contextlib
import sys
import warnings

import pandas as pd

from . import calibrate, clean, eda, evaluate, interpret, loader, paths, schema, split
from .config import Config, load_config
from .models import zoo
from .models.baselines import results_table, run_baselines
from .report import to_markdown


def _load(config: Config, *, allow_synthetic: bool = True) -> loader.Dataset:
    dataset = loader.load_analysis(
        n_synthetic=config.synthetic_rows, seed=config.seed, allow_synthetic=allow_synthetic
    )
    print(f"data: {dataset.describe()}")
    return dataset


def cmd_synth(config: Config, args: argparse.Namespace) -> int:
    """Write a synthetic analysis table to data/interim for inspection."""
    from . import datagen

    paths.ensure_dirs()
    frame = datagen.generate_frame(n_rows=args.rows or config.synthetic_rows, seed=config.seed)
    out = paths.INTERIM / "synthetic_ECRIDataSet.csv"
    frame.to_csv(out, index=False)
    labelled = frame[frame[config.target].notna()]
    print(f"wrote {paths.rel(out)}: {frame.shape[0]:,} rows x {frame.shape[1]} cols")
    print(
        f"  labelled: {len(labelled):,} ({len(labelled) / len(frame):.1%}), "
        f"positive rate {labelled[config.target].mean():.3f}"
    )
    return 0


def cmd_check(config: Config, args: argparse.Namespace) -> int:
    """Validate whatever data is available against the contract."""
    dataset = _load(config)
    loader.validate(dataset.frame)
    frame, report = clean.prepare(
        dataset.frame, config.target, max_missing_rate=config.max_missing_rate
    )
    print(report.summary())
    X, y, groups = split.features_and_target(frame, config.target)
    print(
        f"contract OK: {X.shape[1]} features, {len(y)} labelled rows, "
        f"{0 if groups is None else groups.nunique()} schools"
    )
    print("leakage guard: passed")
    return 0


def cmd_eda(config: Config, args: argparse.Namespace) -> int:
    dataset = _load(config)
    result = eda.run(dataset.frame, config.target, source=dataset.source)
    print(f"\n{result.target_balance.to_string()}\n")
    print("top associations:")
    print(result.associations.head(8).to_string())
    print(f"\nwrote reports/eda.md and {len(result.figures)} figure(s)")
    return 0


def cmd_baseline(config: Config, args: argparse.Namespace) -> int:
    dataset = _load(config)
    frame, report = clean.prepare(
        dataset.frame, config.target, max_missing_rate=config.max_missing_rate
    )
    print(report.summary())

    repeats = max(1, args.repeats)
    collected: dict[str, list[float]] = {}
    tables: list[pd.DataFrame] = []
    for i in range(repeats):
        seed = config.seed + i
        current = split.make_split(
            frame,
            config.target,
            strategy=config.split.strategy,
            test_size=config.split.test_size,
            seed=seed,
        )
        if i == 0:
            print(current.describe())
        results = run_baselines(current, seed=seed)
        tables.append(results_table(results))
        for r in results:
            collected.setdefault(r.name, []).append(r.metrics.roc_auc)
            if r.warning:
                print(f"  WARNING [{r.name}] {r.warning}")

    shown = [
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "recall_at_risk",
        "brier",
    ]
    if repeats == 1:
        print(f"\n{tables[0][shown].round(3).to_string()}")
    else:
        summary = pd.DataFrame(
            {
                "auc_mean": {k: pd.Series(v).mean() for k, v in collected.items()},
                "auc_sd": {k: pd.Series(v).std() for k, v in collected.items()},
                "auc_min": {k: min(v) for k, v in collected.items()},
                "auc_max": {k: max(v) for k, v in collected.items()},
            }
        ).sort_values("auc_mean", ascending=False)
        print(f"\nROC-AUC over {repeats} {config.split.strategy}-splits:")
        print(summary.round(3).to_string())

    paths.ensure_dirs()
    out = paths.REPORTS / "baselines.md"
    out.write_text(
        f"# Baselines\n\nSource: **{dataset.source}** data. Target: `{config.target}`. "
        f"Split: `{config.split.strategy}`, {repeats} repeat(s).\n\n"
        + to_markdown(tables[0][shown].round(3))
        + "\n\n(First split shown; see stdout for the multi-split spread.)\n"
    )
    print(f"\nwrote {paths.rel(out)}")
    return 0


def _prepared(config: Config) -> tuple[pd.DataFrame, loader.Dataset]:
    dataset = _load(config)
    frame, report = clean.prepare(
        dataset.frame, config.target, max_missing_rate=config.max_missing_rate
    )
    print(report.summary())
    return frame, dataset


def _model_names(requested: str | None) -> list[str]:
    available = zoo.available_models()
    if not requested:
        names = list(available)
    else:
        names = [n.strip() for n in requested.split(",") if n.strip()]
        unknown = [n for n in names if n not in zoo.BY_NAME]
        if unknown:
            raise SystemExit(f"unknown model(s): {unknown}; known: {sorted(zoo.BY_NAME)}")
        skipped = [n for n in names if n not in available]
        if skipped:
            print(f"  skipping (optional package absent): {', '.join(skipped)}")
        names = [n for n in names if n in available]
    if not names:
        raise SystemExit("no models available to run")
    return names


def cmd_cv(config: Config, args: argparse.Namespace) -> int:
    """Cross-validate several models on one shared fold plan and compare them."""
    frame, dataset = _prepared(config)
    names = _model_names(args.models)
    print(
        f"cross-validating {len(names)} model(s): {config.split.n_folds} folds x "
        f"{args.repeats} repeats, {config.split.strategy}-split"
    )

    results, plan = evaluate.run_models(
        frame,
        config.target,
        names,
        strategy=config.split.strategy,
        n_folds=config.split.n_folds,
        repeats=args.repeats,
        seed=config.seed,
    )
    table = evaluate.results_table(results)
    print(f"\n{table.round(3).to_string()}")

    for note in evaluate.warnings_for(results):
        print(f"  WARNING {note}")

    reference = args.reference if args.reference in results else table.index[-1]
    comparisons = evaluate.compare_to_reference(
        results, reference, n_boot=args.bootstrap, seed=config.seed
    )
    print(f"\npaired against `{reference}`, 95% bootstrap over students:")
    for comparison in comparisons:
        print(f"  {comparison.summary()}")

    decisive = [c for c in comparisons if c.decisive]
    print(
        f"\n{len(decisive)} of {len(comparisons)} difference(s) clear noise." if comparisons else ""
    )

    paths.ensure_dirs()
    out = paths.REPORTS / "model_comparison.md"
    lines = [
        "# Model comparison",
        "",
        f"Source: **{dataset.source}** data. Target: `{config.target}`. "
        f"Split: `{config.split.strategy}`, {config.split.n_folds} folds x "
        f"{args.repeats} repeats.",
        "",
        "AUC is computed on out-of-fold predictions pooled per repeat; "
        "`auc_sd` is the spread across repeats.",
        "",
        to_markdown(table.round(3)),
        "",
        f"## Paired against `{reference}`",
        "",
        "Bootstrap over students, folds held fixed. A difference is only a "
        "ranking when the interval excludes zero.",
        "",
        *[f"- {c.summary()}" for c in comparisons],
        "",
    ]
    out.write_text("\n".join(lines))
    print(f"wrote {paths.rel(out)}")
    return 0


def cmd_impute(config: Config, args: argparse.Namespace) -> int:
    """Compare imputation strategies for one estimator on shared folds."""
    frame, dataset = _prepared(config)
    print(f"comparing imputation strategies under `{args.base}`")
    results, _ = evaluate.compare_imputation(
        frame,
        config.target,
        base=args.base,
        strategy=config.split.strategy,
        n_folds=config.split.n_folds,
        repeats=args.repeats,
        seed=config.seed,
    )
    table = evaluate.results_table(results)
    print(f"\n{table.round(3).to_string()}")

    reference = "median+indicator"
    comparisons = evaluate.compare_to_reference(
        results, reference, n_boot=args.bootstrap, seed=config.seed
    )
    print(f"\npaired against `{reference}`:")
    for comparison in comparisons:
        print(f"  {comparison.summary()}")
    decisive = [c for c in comparisons if c.decisive]
    verdict = "matters here" if decisive else "does not matter here"
    print(
        f"\n{len(decisive)} of {len(comparisons)} difference(s) clear noise — "
        f"imputation choice {verdict}."
    )

    paths.ensure_dirs()
    out = paths.REPORTS / "imputation.md"
    out.write_text(
        f"# Imputation comparison\n\nSource: **{dataset.source}** data. "
        f"Estimator: `{args.base}`. Target: `{config.target}`. "
        f"{config.split.n_folds} folds x {args.repeats} repeats, "
        f"`{config.split.strategy}`-split.\n\n"
        + to_markdown(table.round(3))
        + f"\n\n## Paired against `{reference}`\n\n"
        + "\n".join(f"- {c.summary()}" for c in comparisons)
        + "\n"
    )
    print(f"wrote {paths.rel(out)}")
    return 0


def cmd_importance(config: Config, args: argparse.Namespace) -> int:
    """Permutation importance on held-out folds, for one or two models."""
    frame, dataset = _prepared(config)
    names = _model_names(args.models)
    results = {}
    for name in names:
        print(f"permuting features under `{name}`...")
        results[name] = interpret.run(
            frame,
            config.target,
            name,
            strategy=config.split.strategy,
            n_folds=config.split.n_folds,
            repeats=args.repeats,
            seed=config.seed,
        )
        table = results[name].table
        print(f"\n{results[name].summary(args.top)}")
        keepers = list(interpret.meaningful(table).index)
        print(f"  above noise ({len(keepers)} of {len(table)}): {', '.join(keepers) or 'none'}")

    if len(results) >= 2:
        first, second = list(results)[:2]
        rho = interpret.agreement(results[first].table, results[second].table)
        print(f"\nrank agreement {first} vs {second}: Spearman rho = {rho:.3f}")

    paths.ensure_dirs()
    out = paths.REPORTS / "importance.md"
    lines = [
        "# Permutation importance",
        "",
        f"Source: **{dataset.source}** data. Target: `{config.target}`. "
        f"Held-out folds, `{config.split.strategy}`-split, "
        f"{config.split.n_folds} folds x {args.repeats} repeats.",
        "",
        "Each number is the drop in held-out ROC-AUC when that raw column is "
        "shuffled. A mean smaller than its SD is noise.",
        "",
    ]
    for name, result in results.items():
        lines += [
            f"## {name}",
            "",
            to_markdown(result.table.round(4)),
            "",
            f"![importance]({result.figure})",
            "",
        ]
    out.write_text("\n".join(lines))
    print(f"wrote {paths.rel(out)}")
    return 0


def cmd_calibrate(config: Config, args: argparse.Namespace) -> int:
    """Reliability curves and a cost-justified threshold for one model."""
    frame, dataset = _prepared(config)
    name = args.model
    if name not in zoo.available_models():
        raise SystemExit(f"model {name!r} is not available")

    results, _ = evaluate.run_models(
        frame,
        config.target,
        [name],
        strategy=config.split.strategy,
        n_folds=config.split.n_folds,
        repeats=args.repeats,
        seed=config.seed,
    )
    result = results[name]
    y_true, scores = result.y_true, result.mean_scores

    calibrators, choice, sweep, figure = calibrate.report(
        y_true, scores, name, cost_ratio=args.cost_ratio
    )
    print(f"\nout-of-fold AUC {result.mean():.3f} ± {result.sd():.3f}")
    print("\ncalibration (fitted and scored on the same out-of-fold predictions,")
    print("so these are an upper bound — they show whether calibration is broken):")
    for cal in calibrators.values():
        print(f"  {cal.method:9s} Brier {cal.brier:.4f}  mean |gap| {cal.calibration_error:.4f}")

    print(f"\nthreshold at {args.cost_ratio:g}:1 (missing an at-risk child : a false flag)")
    print(f"  {choice.summary()}")
    at_default = calibrate.metrics_at(y_true, scores, 0.5)
    at_chosen = calibrate.metrics_at(y_true, scores, choice.threshold)
    print(
        f"  at 0.50          recall(at-risk) {at_default.recall_at_risk:.3f}  "
        f"precision {1 - at_default.fp / max(1, at_default.fp + at_default.tn):.3f}"
    )
    print(f"  at {choice.threshold:.2f}          recall(at-risk) {at_chosen.recall_at_risk:.3f}")
    print(f"\nhow the threshold moves with the cost assumption:\n{sweep.to_string()}")

    paths.ensure_dirs()
    out = paths.REPORTS / "calibration.md"
    out.write_text(
        f"# Calibration and threshold — `{name}`\n\n"
        f"Source: **{dataset.source}** data. Target: `{config.target}`. "
        f"Out-of-fold over {config.split.n_folds} folds x {args.repeats} repeats, "
        f"`{config.split.strategy}`-split. AUC {result.mean():.3f} ± {result.sd():.3f}.\n\n"
        "## Calibration\n\nFitted and scored on the same out-of-fold predictions, so "
        "optimistic by construction. They answer whether calibration is broken, not "
        "what the Brier score is.\n\n"
        + to_markdown(
            pd.DataFrame(
                [
                    {
                        "method": c.method,
                        "brier": round(c.brier, 4),
                        "mean_abs_gap": round(c.calibration_error, 4),
                    }
                    for c in calibrators.values()
                ]
            ).set_index("method")
        )
        + f"\n\n![reliability]({figure})\n\n## Threshold\n\n"
        f"Chosen at a stated cost ratio of {args.cost_ratio:g}:1 — missing a child who "
        "falls short against a second look at a child who was fine. This is an "
        "assumption, not a measurement.\n\n"
        f"- {choice.summary()}\n\n" + to_markdown(sweep) + "\n"
    )
    print(f"\nwrote {paths.rel(out)}")
    return 0


COMMANDS = {
    "synth": cmd_synth,
    "check": cmd_check,
    "eda": cmd_eda,
    "baseline": cmd_baseline,
    "cv": cmd_cv,
    "impute": cmd_impute,
    "importance": cmd_importance,
    "calibrate": cmd_calibrate,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ecri", description=__doc__)
    parser.add_argument("--config", help="path to a YAML config (default: configs/default.yaml)")
    parser.add_argument("--target", choices=schema.TARGETS, help="override the configured target")
    parser.add_argument("--strategy", choices=["school", "student"], help="override split strategy")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress warnings")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="validate the data against the contract")
    sub.add_parser("eda", help="write reports/eda.md and figures")
    p_synth = sub.add_parser("synth", help="write a synthetic analysis table")
    p_synth.add_argument("--rows", type=int, default=0)
    p_base = sub.add_parser("baseline", help="fit and score the baselines")
    p_base.add_argument("--repeats", type=int, default=1, help="repeat over N random splits")

    p_cv = sub.add_parser("cv", help="cross-validate and compare models")
    p_cv.add_argument("--models", help="comma-separated names (default: every available model)")
    p_cv.add_argument("--repeats", type=int, default=5, help="independent fold plans")
    p_cv.add_argument("--reference", default="logistic", help="model to compare the rest against")
    p_cv.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples")

    p_imp = sub.add_parser("impute", help="compare imputation strategies")
    p_imp.add_argument("--base", default="logistic", help="estimator to hold fixed")
    p_imp.add_argument("--repeats", type=int, default=5)
    p_imp.add_argument("--bootstrap", type=int, default=2000)

    p_fi = sub.add_parser("importance", help="permutation importance on held-out folds")
    p_fi.add_argument("--models", default="logistic", help="comma-separated names")
    p_fi.add_argument("--repeats", type=int, default=3)
    p_fi.add_argument("--top", type=int, default=12, help="rows to print")

    p_cal = sub.add_parser("calibrate", help="reliability curves and threshold selection")
    p_cal.add_argument("--model", default="logistic")
    p_cal.add_argument("--repeats", type=int, default=5)
    p_cal.add_argument(
        "--cost-ratio",
        type=float,
        default=calibrate.DEFAULT_COST_RATIO,
        help="cost of missing an at-risk child vs a false flag",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    overrides = {}
    if args.target:
        overrides["target"] = args.target
    if args.strategy:
        from dataclasses import replace

        overrides["split"] = replace(config.split, strategy=args.strategy)
    if overrides:
        from dataclasses import replace

        config = replace(config, **overrides)

    # catch_warnings scopes --quiet to this call. A bare simplefilter() would
    # mutate the process-wide filter and silence an importing caller's warnings
    # too, which is how the synthetic-data warning went missing under pytest.
    with contextlib.ExitStack() as stack:
        if args.quiet:
            stack.enter_context(warnings.catch_warnings())
            warnings.simplefilter("ignore")
        return COMMANDS[args.command](config, args)


if __name__ == "__main__":
    sys.exit(main())
