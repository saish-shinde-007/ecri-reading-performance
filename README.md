# Predicting Early Childhood Reading Performance

Machine learning on the **Enhanced Core Reading Instruction (ECRI)** dataset
([Shangguan et al., NeurIPS 2024 Datasets & Benchmarks][paper]) — predicting
which at-risk early readers will make adequate growth, from what is known about
them at baseline.

A screening aid for allocating reading support earlier. Not a placement
decision, and not a replacement for a teacher's judgement.

[![CI](https://github.com/saish-shinde-007/ecri-reading-performance/actions/workflows/ci.yml/badge.svg)](https://github.com/saish-shinde-007/ecri-reading-performance/actions/workflows/ci.yml)

---

## The task, stated precisely

The dataset ships two binary `*_target` columns and does not document them.
Reverse-engineering them changed the shape of the project, so start here:

> **`RMwidRS_target = 1` ⟺ Woodcock Word-ID gain from `t=0` to `t=1` ≥ 27**
> — reproduces on **946 / 946** students, exactly.
> `RMwdaRS_target` is the same rule at a threshold of 11.

Run `make verify` to check it yourself. Consequences, in order of importance:

1. The task is **responder prediction** — who will grow adequately — not score
   prediction. That is a more useful question for deciding where support goes.
2. It is **classification**, settling a question the proposal left open.
3. The **leakage boundary is exact**: any `t ≥ 0.5` measurement reconstructs
   the label. Baseline `RMwidRS` is a legal feature; `RMwidRS` at `t=1` is a
   perfect leak. `ecri.split.assert_no_leakage()` enforces this and raises
   rather than warns.

Full write-up, with five further measured findings: **[docs/findings.md](docs/findings.md)**.

## Results so far

Logistic regression, ROC-AUC on held-out data, mean ± SD over 12 random splits:

| Model | Word ID, school-split | Word ID, student-split | Word Attack, school-split |
|---|---|---|---|
| **Logistic regression, all features** | **0.657 ± 0.037** | 0.692 ± 0.034 | 0.611 ± 0.037 |
| Logistic, baseline Woodcock score only | 0.602 ± 0.029 | 0.592 ± 0.028 | 0.507 ± 0.049 |
| Decision tree, depth 3 | 0.593 ± 0.021 | 0.613 ± 0.027 | 0.523 ± 0.021 |
| Majority class | 0.500 | 0.500 | 0.500 |

Read these carefully:

- **The spread is the result.** One split gave 0.703 — the top of a 0.594–0.720
  range. With ~200 test rows the SD is ~0.04, so single numbers mean nothing
  here, and `--repeats` exists for that reason.
- **0.66 is a real result, not a weak one.** The strongest single feature
  correlates at |r| = 0.20. Anything near 0.9 on this data is a leak, and
  `metrics.leakage_warning()` says so at 0.85.
- **School-split is the honest number.** Classroom and teacher features are
  constant within a classroom, so a student-level split puts the same values on
  both sides. A quoted AUC without its split strategy is not interpretable.
- On **Word Attack** the baseline score alone is at chance (0.507) while the
  full feature set reaches 0.611 — the signal there is entirely in the
  classroom, teacher and other-assessment features.

## Quick start

```bash
make install     # uv venv + editable install
make check       # validate the contract, run the leakage guard
make eda         # reports/eda.md + 4 figures
make baseline    # baselines over 8 school-splits
```

No data needed — everything falls back to a synthetic generator built from the
real file's aggregate statistics, and says so loudly on every run. For real
numbers:

```bash
make data        # fetch the three CSVs into data/raw (CC BY-NC-SA 4.0)
make verify      # confirm the label rule still reproduces
make all
```

## How it fits together

```
data/raw/ECRIDataSet.csv ──┐
   (or ecri.datagen) ──────┴─► loader.load_analysis   validate the contract
                                      │                duplicate StuID → error
                                      ▼                post-baseline col → error
                               clean.prepare          6,803 → 946 rows
                                      │                label filter BEFORE the
                                      ▼                missingness gate
                            split.make_split          whole schools held out
                                      │                assert_no_leakage
                                      ▼
                          features.build_preprocessor  impute + indicators,
                                      │                fit on train only
                                      ▼
                             models.run_baselines  ──► metrics.evaluate
```

| Module | Responsibility |
|---|---|
| [`schema.py`](src/ecri/schema.py) | The data contract: column roles and the leakage deny-list |
| [`datagen.py`](src/ecri/datagen.py) | Synthetic ECRI-shaped data from measured aggregates |
| [`loader.py`](src/ecri/loader.py) | Load real-or-synthetic, validate, record provenance |
| [`clean.py`](src/ecri/clean.py) | Label filter, missingness gate, range flags |
| [`split.py`](src/ecri/split.py) | School-disjoint splits and the leakage guard |
| [`features.py`](src/ecri/features.py) | Impute, encode, scale, indicate missingness |
| [`metrics.py`](src/ecri/metrics.py) | Classification metrics + at-risk recall |
| [`models/baselines.py`](src/ecri/models/baselines.py) | The five baselines |
| [`eda.py`](src/ecri/eda.py) | Regenerates every number in findings.md |
| [`cli.py`](src/ecri/cli.py) | `ecri check / synth / eda / baseline` |

Three decisions worth knowing before reading the code:

**The label filter runs before the missingness gate.** `ORFwc` is 78.8% missing
over all rows but 1.1% on the labelled rows. Gating first would delete the
project's better features for a reason that does not apply to the rows being
modelled.

**Missingness is a feature.** The masks are block-structured — `TKPctCrct` and
`TYearsTeaching` vanish together as one unreturned teacher survey — so
`add_indicator=True` keeps "was this measured" rather than imputing it away.

**Identifiers are never features.** `SchlID` and `TeachID` are
small-cardinality integers a tree will happily memorise into a classroom-level
label lookup. The guard rejects them.

## Project status

**Phase 1 complete** — 16 tickets, 124 tests, 96% coverage, CI on 3.11/3.12/3.13.
**Phase 2** — 22 tickets planned in **[PLAN.md](PLAN.md)**, starting with the
repeated-CV harness, because every model comparison after this point is noise
without it.

## Data and licence

Code is MIT. **The ECRI data is CC BY-NC-SA 4.0 and is not in this repo** —
share-alike would reach the code sitting next to it. `make data` fetches it;
`data/` is git-ignored. Cite Shangguan et al. (2024) in anything derived from
it. See [docs/data.md](docs/data.md). This repo asserts no rights over the data.

[paper]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/862170d83a17b7c2ef3c5a58e7f1992d-Abstract-Datasets_and_Benchmarks_Track.html
