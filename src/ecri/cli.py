"""Command line entry point: `ecri {synth,eda,baseline,check}`."""

from __future__ import annotations

import argparse
import contextlib
import sys
import warnings

import pandas as pd

from . import clean, eda, evaluate, loader, paths, schema, split
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


COMMANDS = {
    "synth": cmd_synth,
    "check": cmd_check,
    "eda": cmd_eda,
    "baseline": cmd_baseline,
    "cv": cmd_cv,
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
