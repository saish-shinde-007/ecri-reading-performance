"""Command line entry point: `ecri {synth,eda,baseline,check}`."""

from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

from . import clean, eda, loader, paths, schema, split
from .config import Config, load_config
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
    print(f"wrote {out.relative_to(paths.ROOT)}: {frame.shape[0]:,} rows x {frame.shape[1]} cols")
    print(f"  labelled: {len(labelled):,} ({len(labelled) / len(frame):.1%}), "
          f"positive rate {labelled[config.target].mean():.3f}")
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
    print(f"contract OK: {X.shape[1]} features, {len(y)} labelled rows, "
          f"{0 if groups is None else groups.nunique()} schools")
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
            frame, config.target, strategy=config.split.strategy,
            test_size=config.split.test_size, seed=seed,
        )
        if i == 0:
            print(current.describe())
        results = run_baselines(current, seed=seed)
        tables.append(results_table(results))
        for r in results:
            collected.setdefault(r.name, []).append(r.metrics.roc_auc)
            if r.warning:
                print(f"  WARNING [{r.name}] {r.warning}")

    shown = ["roc_auc", "average_precision", "accuracy", "balanced_accuracy", "f1",
             "recall_at_risk", "brier"]
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
    print(f"\nwrote {out.relative_to(paths.ROOT)}")
    return 0


COMMANDS = {
    "synth": cmd_synth,
    "check": cmd_check,
    "eda": cmd_eda,
    "baseline": cmd_baseline,
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quiet:
        warnings.simplefilter("ignore")

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

    return COMMANDS[args.command](config, args)


if __name__ == "__main__":
    sys.exit(main())
