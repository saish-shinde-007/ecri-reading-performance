# Findings

Things established by running against the real data. Each one overrides
`docs/proposal.md` where they disagree.

## F1 — The labels are adequate-growth flags, and the rule is exact

`RMwidRS_target` and `RMwdaRS_target` are not thresholds on a score *level*.
Each is a threshold on the **gain** in that Woodcock measure between the
baseline time point (`t=0`) and the next one (`t=1`):

| Label | Rule | Agreement |
|---|---|---|
| `RMwidRS_target` | `RMwidRS@t1 − RMwidRS@t0 ≥ 27` | **946 / 946 = 100.00%** |
| `RMwdaRS_target` | `RMwdaRS@t1 − RMwdaRS@t0 ≥ 11` | **946 / 946 = 100.00%** |

Reproduce with `python tools/verify_target_rule.py` (needs the real CSVs).
The search is over every observed gain value for a single `gain >= threshold`
rule, so an exact hit is not a fitted artefact — one cut point separates the
classes with zero errors in both cases.

Why it matters:

1. **It closes the proposal's "confirm the prediction target" question** with
   proof instead of an assumption.
2. **It makes the task a responder-prediction problem**, not a score-prediction
   one: who will respond adequately to instruction, given only what is known at
   baseline. That is a more useful question for allocating intervention, and it
   is the one the paper actually poses.
3. **It fixes the leakage boundary exactly.** Any `t ≥ 0.5` measurement from
   `ECRIRawData.csv` is disqualified, and the `t=1` Woodcock score is a perfect
   leak. Baseline `RMwidRS`/`RMwdaRS` in the analysis table are legitimate
   features — they are the `t=0` term of the gain, and on their own they are
   weak predictors (see F3).
4. **The task is framed as classification, not regression.** The proposal left
   the choice open pending the labels; the labels are binary, so classification
   it is. Regression on raw gain remains available from the raw file, and is
   deferred (PLAN T22).

## F2 — The "longitudinal" analysis table is a single time point

`ECRIDataSet.csv` has one row per student (6,803 unique `StuID` in 6,803 rows)
with `Time` ≡ 1.0 and `t` ≡ 0.0. The repeated measurements live in
`ECRIRawData.csv` (6,916 students × 4 time points). So the modelling table is
cross-sectional; the longitudinal structure is entirely inside the label and in
the raw file. Any code that groups the analysis table by student and expects
several rows is wrong.

`Keep` is also constant (1.0 everywhere), so it is not a usable filter —
the filtering it records already happened.

## F3 — Labels exist for 946 students, nearly all Tier 2, and the signal is weak

Only 946 of 6,803 rows (13.9%) carry labels — exactly the rows that also have
baseline Woodcock scores and `Age1b`. Of those, **98.4% are Tier 2**
(931 of 946; 9 Tier 1, 6 Tier 3). Labels are spread over all 44 schools and
143 teachers.

So the modelling population is "at-risk readers receiving Tier-2 support", not
all students. Both treatment arms are present (517 control, 429 intervention).

Baseline predictors are only weakly associated with the label:

| Feature | point-biserial r with `RMwidRS_target` |
|---|---|
| `RMwidRS` (baseline Word ID) | −0.200 |
| `SAsrS` | −0.132 |
| `ORFwc` | −0.132 |
| `SAtoS` | −0.071 |
| `Tier` | +0.077 |
| `Tier2_N` | +0.074 |
| `gnrl_fid` | +0.065 |
| `SAwrS` | −0.042 |
| `NWFcls` | −0.011 |

The strongest single association is |r| = 0.20, so no single feature is a
useful predictor. Combining them does better than that ceiling suggests — see
F7 for what baselines actually reach — but nothing here supports a 0.9-AUC
expectation, and `metrics.leakage_warning()` treats anything at or above 0.85
as a leak to investigate rather than a result to report.

The negative sign on baseline score is coherent with F1 rather than surprising:
lower starters have more headroom to gain 27 points, so a gain-defined label is
mildly *anti*-correlated with starting level.

## F4 — Class balance is fine; the proposal's imbalance concern does not apply here

`RMwidRS_target`: 499 positive / 447 negative (52.7% positive).
`RMwdaRS_target`: 545 / 401 (57.6% positive).

Near-balanced, so class weights and resampling are not needed for these two
targets. The imbalance work in the proposal is not wasted, but it is deferred
and conditional (PLAN T24) — it becomes relevant only if the target is
redefined at a stricter threshold.

## F5 — Missingness is severe and structured, and far milder on the labelled rows

Null rates over all 6,803 rows:

| Column | Null % | Note |
|---|---|---|
| `Age1b`, `RMwidRS`, `RMwdaRS`, both targets | 86.1% | the same 5,857 rows — Woodcock was administered to a subsample |
| `ORFwc` | 78.8% | |
| `NWFcls`, `NWFwrc` | 56.5% | identical masks |
| `grp_rate` | 42.0% | classroom-level, missing for whole classrooms |
| `rcmistot` | 40.2% | classroom-level |
| `Tier2MinRd` | 25.8% | |
| `TKPctCrct`, `TYearsTeaching` | 24.7% | identical masks — teacher survey non-response |
| `Tier` | 11.5% | |
| `SAwrS`/`SAsrS`/`SAtoS` | ~11% | |

**But those rates are the wrong ones to design against.** On the 946 labelled
rows — the only rows that can be trained or evaluated on — missingness largely
disappears, because the students given the Woodcock subsample assessment were
also the ones measured on everything else:

| Column | Null % on labelled rows |
|---|---|
| `Tier2MinRd` | 24.8% |
| `TKPctCrct`, `TYearsTeaching` | 22.3% |
| `grp_rate` | 5.6% |
| `rcmistot` | 3.5% |
| `gnrl_fid` | 3.2% |
| `NWFcls`, `NWFwrc`, `ORFwc` | 1.1% |
| `Age1b` | 0.4% |
| `SpEd` | 0.4% |
| `LEP` | 0.3% |
| everything else, `Tier` included | 0.0% |

`ORFwc` goes from 78.8% missing to 1.1%. A missingness gate applied before
filtering to labelled rows would throw away the project's best features, so
`max_missing_rate` is applied **after** the label filter (T07) and the default is
deliberately loose at 0.95.

Two further consequences. The surviving masks come in blocks — `TKPctCrct` and
`TYearsTeaching` are missing together because a teacher did not return the
survey, not because two values independently went astray — so missingness is
informative and worth encoding as indicator columns (T10) rather than only
imputed away. And the remaining gaps are small enough that imputation strategy
is unlikely to dominate results, which lowers the priority of the imputation
comparison (T23).

## F6 — Counts do not match the paper's headline

The paper and project page say 6,916 students and 172 teachers. The analysis
table has 6,803 students and 171 distinct `TeachID`, with 81 rows missing
`TeachID` entirely. The raw file does have 6,916. So 113 students are dropped
between the two files and the published code's own filtering is not documented.
Use the raw file's counts when quoting the dataset, the analysis table's when
quoting this project's n.

## F7 — Baselines: logistic regression reaches 0.66 AUC, and the spread matters

Measured with `make baseline`, 12 random splits per cell, mean ± SD of ROC-AUC
on held-out data:

### `RMwidRS_target` (Word Identification)

| Model | School-split | Student-split |
|---|---|---|
| logistic regression, all features | **0.657 ± 0.037** | 0.692 ± 0.034 |
| logistic, baseline Woodcock score only | 0.602 ± 0.029 | 0.592 ± 0.028 |
| decision tree, depth 3 | 0.593 ± 0.021 | 0.613 ± 0.027 |
| stratified guess | 0.513 ± 0.032 | 0.483 ± 0.033 |
| majority class | 0.500 | 0.500 |

### `RMwdaRS_target` (Word Attack)

| Model | School-split | Student-split |
|---|---|---|
| logistic regression, all features | **0.611 ± 0.037** | 0.638 ± 0.043 |
| decision tree, depth 3 | 0.523 ± 0.021 | 0.541 ± 0.021 |
| logistic, baseline score only | 0.507 ± 0.049 | 0.501 ± 0.045 |
| stratified guess | 0.516 ± 0.026 | 0.503 ± 0.027 |
| majority class | 0.500 | 0.500 |

Four things to carry forward:

**The spread is the headline, not the mean.** A single school-split put logistic
regression at 0.703 on Word ID — the top of a 0.594–0.720 range. With ~200 test
rows the per-split SD is ~0.037, so any two models within ~0.07 of each other on
one split are indistinguishable. **Report mean ± SD over repeated splits, never a
single number.** This is the main reason the phase-2 CV harness (T21) comes
before any model tuning rather than after.

**School-split is harder than student-split**, by 0.03–0.04 on both targets, as
it should be: a held-out school shares no teacher, classroom or fidelity context
with training. The default strategy stays `school`, and a result quoted without
its split strategy is not interpretable.

**Multivariate signal is real and the univariate correlations understate it.**
F3's best single predictor is |r| = 0.20, equivalent to roughly 0.61 AUC —
which is what the baseline-score-only model gets on Word ID. The full feature
set adds ~0.055 on top. On Word Attack the baseline score alone is at chance
(0.507) while the full set reaches 0.611, so for that target essentially all the
signal is in the classroom, teacher and other-assessment features.

**Word Attack is the harder target.** ~0.046 lower AUC, and no usable
single-feature signal. `RMwidRS_target` is the better primary task and is the
config default.

Bottom line: a phase-2 model is only interesting if it clears **0.657 ± 0.037**
school-split on Word ID over repeated splits. That is the bar.
