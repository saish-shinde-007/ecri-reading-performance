# Reproducibility

Every number this project reports depends on three things only: the data, the
software environment, and the seed. Each is pinned, and `make reproduce`
re-derives the headline numbers to show nothing else has crept in.

```bash
make install         # the exact environment in uv.lock
make data            # the exact bytes, verified
make reproduce       # re-derive and compare with results/expected.json
make release-check   # all of it, plus lint and tests — the pre-tag gate
```

## 1. Data: pinned to a commit and verified by sha256

`make data` downloads from commit `7168d48` of
[ecri-data/ECRI-Dataset](https://github.com/ecri-data/ECRI-Dataset) (the
repo's last commit, 2024-08-19), not from `main`. It refuses any file whose
sha256 differs from the pin:

| File | sha256 (first 16) |
|---|---|
| `ECRIDataSet.csv` | `1883c5796132cfb5` |
| `ECRIRawData.csv` | `ca214ae9397b313b` |
| `MetaData.csv` | `aae2184080c30b7b` |

The pins are in `src/ecri/sources.py`. `python tools/fetch_data.py --check`
verifies local copies without touching the network. The reproduction gate
checks them before computing anything, so a data problem reports itself as a
data problem rather than as a changed result.

## 2. Environment: locked

`uv.lock` pins every package, direct and transitive. `make install` runs
`uv sync --locked`, which installs exactly the lock and fails if
`pyproject.toml` has changed without re-locking. CI installs the same way.

The canonical interpreter is **Python 3.12** (`.python-version`). All recorded
numbers were produced with:

| Package | Version |
|---|---|
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |

On Python 3.11 the lock forks to numpy 2.4.6 and scipy 1.17.1, because numpy
2.5 dropped 3.11. CI tests that fork, but no recorded number comes from it.

## 3. Seeds: audited, and the audit is a test

Every place randomness enters, and where its seed comes from
(`config.seed` = 20260101):

| Where | Seed |
|---|---|
| CV fold plans (`split.cv_folds`) | `config.seed + repeat` |
| single train/test split (`split.make_split`) | `config.seed` |
| every estimator (`models/zoo.py`) | `config.seed + 100·repeat + fold` |
| paired bootstrap (`evaluate.paired_bootstrap`) | `config.seed` |
| permutation importance (`interpret`) | `config.seed + fold` |
| synthetic data (`datagen`) | `config.seed`, or the caller's |
| model probe (`persist`) | `PROBE_SEED` = 424242 |
| iterative imputer | 0 |
| Platt calibration | none needed: a deterministic solver |

`tests/test_determinism.py` holds the audit in place:

- a **static scan** fails the build on any unseeded randomness: `default_rng()`,
  numpy's global state, `random_state=None`, or the stdlib `random` module;
- the **same seed gives bit-identical** out-of-fold scores for every estimator
  family, and a different seed gives different folds;
- results are **independent of `PYTHONHASHSEED`**, checked across processes, so
  nothing depends on set or dict iteration order.

The audit found one problem. The random forest was not bit-reproducible with
`n_jobs=2`: two runs with the same seed differed by 1.1e-16. scikit-learn sums
the trees' votes in whatever order the threads finish, and floating-point
addition isn't associative. No reported number moved, since AUC agreed to 12
decimals, but every estimator now runs single-threaded so that the same inputs
give the same output, exactly.

## 4. The gate: `make reproduce`

`ecri reproduce` re-derives, from the pinned data:

- the **label rule** (F1): both thresholds, and that they hold exactly;
- the **nine-model comparison** at 10 repeats (F8): every model's AUC, SD,
  average precision, at-risk recall and Brier score, plus every paired
  interval against logistic regression and its verdict;
- the **deployed model's** cross-validated AUC (T27).

It compares each value with `results/expected.json`, the committed record:

| Situation | Result |
|---|---|
| a number moves by more than 0.0005 | **fail**: names the number, old → new |
| a "clears noise" verdict flips | **fail**, even if the numbers stay close |
| a recorded number is missing | **fail** |
| a number not in the record appears | reported, not failed |
| a thresholded count (at-risk recall) moves, on a **different CPU architecture** from the record | reported, not failed (§5) |
| the canonical configuration changed | refused: the numbers aren't comparable |
| the data differs from the pins | refused before anything runs |

`ecri reproduce --write` re-records deliberately and lists what moved.
RELEASING.md says when that is legitimate: never without a written reason in
`docs/findings.md`.

**The README results table is generated** from the same record by
`ecri readme`, and `tests/test_reproduce.py` fails if the README and the record
ever disagree. They already had: after the F14 re-run, the hand-typed table
still showed the MLP at 0.617 ± 0.028 (it is 0.620 ± 0.026), and four
intervals were 0.001 stale.

## 5. Where it has been reproduced

Each run is a fresh `git clone`, then `make install`, `make data`, `make
release-check` and every target of `make all`. The record was made on the M4.

| Machine | CPU | Python | Gated numbers | Largest difference |
|---|---|---|---|---|
| Compute box (M4, 12 cores) | arm64, macOS | 3.12.3 | 85 of 85 agree | **0.0** |
| Laptop (M1 Air, 8 cores) | arm64, macOS | 3.12.9 | 85 of 85 agree | **0.0** |
| CI runner (GitHub Actions) | x86-64, Linux | 3.12 | 84 of 85 agree, then 84 of 84 under the rule below | ≤ 5e-4 |

**On ARM it is bit-exact.** The laptop and the M4 differ in chip, core count
and Python patch version, yet every recorded number matches to the last
stored digit.

**On x86-64 one number moved:** the random forest's at-risk recall, by
+6.7e-4. Everything continuous held: every AUC, SD, average precision and
Brier score, and every paired interval and verdict. The cause is how the
forest gets built, not how it predicts:

- Tree construction picks the best split by comparing impurity
  improvements. When two candidate splits nearly tie, the different rounding
  of intermediate sums on x86-64 and on ARM can pick different ones. A
  changed tree moves a student's score by up to 1/500, one tree's worth.
- At-risk recall is a count of students below a fixed cut of 0.5. On the
  real data, 29 at-risk student-repeats have a forest score within 0.001 of
  that cut, and 145 within 0.005.
- One flip moves the 10-repeat recall by 0.000224. The observed +6.7e-4 is
  three flips.

AUC barely notices, because it depends on rank order, not on which side of a
cut a score lands. So **thresholded counts are gated on the architecture the
record was made on**, where they are bit-exact, and **reported elsewhere**.
Every conclusion the project draws rests on the continuous numbers, and those
are gated everywhere. To make the record on x86-64 instead, re-record there
with `--write`.

## 6. How long it takes

From a fresh clone, uv's package cache already warm:

| Step | Laptop (M1, 8 cores) | M4 (12 cores) |
|---|---|---|
| `make install` | 3 s | 1 s |
| `make data` | under 1 s | 1 s |
| `make release-check` (lint, 260 tests, checksums, label rule, README check, reproduction) | 414 s | 230 s |
| `make all` | **331 s** | **202 s** |
| &nbsp;&nbsp;check · eda · baseline | 6 · 6 · 8 s | 1 · 2 · 2 s |
| &nbsp;&nbsp;cv (9 models, 5 repeats) | 87 s | 64 s |
| &nbsp;&nbsp;impute · calibrate · school | 30 · 7 · 24 s | 18 · 3 · 13 s |
| &nbsp;&nbsp;importance (logistic + random forest) | 140 s | 96 s |
| &nbsp;&nbsp;model (train, save, verify) | 23 s | 3 s |

`make all` regenerates all seven reports in `reports/`, seven figures, and the
saved model. The reproduction step on its own took 108 s on the M4 and about
3 minutes on a GitHub Actions runner.
