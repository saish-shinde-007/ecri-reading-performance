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

<!-- results:start -->
9 models, 5-fold school-disjoint CV repeated 10 times, ROC-AUC on out-of-fold predictions, Word Identification target. Intervals are a paired bootstrap over students with the folds held fixed (4,000 resamples).

| Model | AUC | vs logistic regression, 95% CI |
|---|---|---|
| **Logistic regression** | **0.662 ± 0.008** | — |
| Random forest | 0.652 ± 0.005 | −0.008 [−0.038, +0.022] · within noise |
| Hist gradient boosting | 0.644 ± 0.008 | −0.006 [−0.037, +0.025] · within noise |
| Hist GB, native NaN | 0.641 ± 0.010 | −0.009 [−0.039, +0.022] · within noise |
| MLP (32-16) | 0.620 ± 0.026 | −0.004 [−0.024, +0.017] · within noise |
| Decision tree, depth 3 | 0.600 ± 0.014 | −0.048 [−0.084, −0.012] · **worse** |
| Logistic, baseline score only | 0.593 ± 0.002 | −0.072 [−0.108, −0.036] · **worse** |
| Stratified guess | 0.505 ± 0.015 | −0.148 [−0.199, −0.096] · **worse** |
| Majority class | 0.500 ± 0.000 | −0.165 [−0.198, −0.130] · **worse** |

*Generated from [`results/expected.json`](results/expected.json) by `ecri readme`. `make reproduce` re-derives every number from the pinned data and fails if any moves by more than 0.0005.*
<!-- results:end -->

What the intervals say:

- **Nothing beats logistic regression.** Every flexible model sits inside the
  noise band, and the MLP's spread is several times logistic's. Logistic is the
  model to ship: the most accurate, the most stable, and explainable to a
  teacher. (F8)
- **An AUC near 0.66 is a real result.** The best single feature correlates at
  |r| = 0.20, and the multivariate model beats the single-feature one
  decisively. Anything near 0.9 here would be a leak. (F3, F7)
- **The useful output is a ranking, not a flag.** With nearly half the
  students falling short and a ranker this strong, the cost-optimal yes/no
  threshold flags nearly everyone once missing a struggling reader costs even
  twice a false alarm. Ordering a fixed support budget is where the model
  earns its keep. (F11)
- **It ranks Tier-2 readers who have a baseline Woodcock score.** For every
  other student in the cohort that score is missing, so it is not a
  whole-school screener. (F15)
- **Five or six features carry it**; baseline Word ID is an order of magnitude
  ahead of everything else (F12). **Imputation choice doesn't matter** (F10),
  and **school racial composition is excluded**: it adds nothing, and a
  support-allocation tool must not use it (F13).

Every finding, with the numbers behind it: **[docs/findings.md](docs/findings.md)**.

## Quick start

```bash
make install     # uv venv + editable install
make check       # validate the contract, run the leakage guard
make eda         # reports/eda.md + figures
make cv          # 9-model comparison, repeated CV, paired bootstrap
make calibrate   # reliability curves + cost-justified threshold
make importance  # permutation importance, two model families
make model       # train, calibrate, save and verify the deployable model
make rank-demo   # rank a batch of students with it, most at risk first
```

To use the model from code — loading it, ranking students, reading its
metadata — see **[docs/inference.md](docs/inference.md)**, the contract the app
and the model card are built on.

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

**66% shipped.** Phase 1 (foundation, 40%), phase 2a (evaluation and models,
20%), and our phase-2b contracts (6%): model persistence (T27) and the ranking
contract (T28). 229 tests, 95% coverage, CI on Python 3.11–3.13.

Building the ranking contract turned up two things the metrics never showed:
a single blank cell could move a student half the queue (F14, fixed), and the
model is grounded only for Tier-2 readers with a baseline Woodcock score —
for every other student in the cohort its most important input is missing
(F15).

**Remaining 34%**, in **[PLAN.md](PLAN.md)**: Sarvesh's tickets
([#2–#11](https://github.com/saish-shinde-007/ecri-reading-performance/issues?q=is%3Aissue%20label%3A%22owner%3A%20sarvesh%22)),
now unblocked, and our release ticket T39.

## Data and licence

Code is MIT. **The ECRI data is CC BY-NC-SA 4.0 and is not in this repo** —
share-alike would reach the code sitting next to it. `make data` fetches it;
`data/` is git-ignored. Cite Shangguan et al. (2024) in anything derived from
it. See [docs/data.md](docs/data.md). This repo asserts no rights over the data.

[paper]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/862170d83a17b7c2ef3c5a58e7f1992d-Abstract-Datasets_and_Benchmarks_Track.html
