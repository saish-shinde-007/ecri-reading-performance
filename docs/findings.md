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

## F8 — No non-linear model beats logistic regression

Nine models, 5 folds × 5 repeats, school-disjoint, out-of-fold AUC pooled per
repeat. Reproduce with `make cv`.

| Model | AUC (mean ± SD) | AP | recall (at-risk) | Brier |
|---|---|---|---|---|
| **logistic regression** | **0.662 ± 0.010** | 0.655 | 0.557 | 0.232 |
| random forest | 0.655 ± 0.006 | 0.643 | 0.468 | 0.233 |
| hist gradient boosting | 0.647 ± 0.005 | 0.652 | 0.540 | 0.241 |
| hist GB, native NaN | 0.644 ± 0.006 | 0.647 | 0.540 | 0.243 |
| MLP (32-16) | 0.618 ± 0.040 | 0.612 | 0.489 | 0.239 |
| decision tree, depth 3 | 0.603 ± 0.010 | 0.608 | 0.369 | 0.247 |
| logistic, baseline score only | 0.592 ± 0.002 | 0.574 | 0.426 | 0.241 |
| stratified guess | 0.511 ± 0.013 | 0.533 | 0.485 | 0.488 |
| majority class | 0.500 | 0.527 | 0.000 | 0.473 |

Paired bootstrap over students against logistic regression, folds held fixed,
95% interval:

| Difference | ΔAUC | 95% CI | Verdict |
|---|---|---|---|
| vs majority | +0.165 | [+0.130, +0.198] | clears noise |
| vs stratified | +0.139 | [+0.089, +0.188] | clears noise |
| vs baseline-score-only | +0.073 | [+0.037, +0.110] | clears noise |
| vs tree depth 3 | +0.042 | [+0.005, +0.080] | clears noise |
| vs hist GB native NaN | +0.008 | [−0.024, +0.038] | within noise |
| vs MLP | +0.008 | [−0.015, +0.031] | within noise |
| vs random forest | +0.005 | [−0.025, +0.037] | within noise |
| vs hist gradient boosting | +0.004 | [−0.029, +0.035] | within noise |

**Four differences clear noise and four do not, and the split between them is
the finding.** Logistic regression beats every baseline decisively — including
the single-feature model by +0.073, so the multivariate signal is real. It then
beats no non-linear model by any margin that survives resampling the students.

Why that is the expected answer rather than a disappointment: 946 rows, 22
features, and a best single correlation of |r| = 0.20 (F3). There is not enough
signal or sample for a flexible model to find structure a linear one misses,
and the MLP's SD of 0.040 — four times logistic's — shows what extra capacity
buys here, which is variance.

Consequences:

1. **Logistic regression is the model to ship.** It is the most accurate, the
   most stable, the cheapest, and the easiest to explain to a teacher or
   administrator. Preferring a tree ensemble on a 0.005 difference would be
   choosing opacity for noise.
2. **The boosting ticket is not a likely win.** XGBoost and LightGBM are the
   same model family as the two gradient-boosting entries already measured
   here; both landed below logistic. The ticket stays in the plan for
   completeness, with the expectation set.
3. **Report the interval, not the ranking.** The table is sorted by AUC, but
   rows 1–4 are one statistical tie. A write-up that calls random forest
   "second best" is reading noise as order.

**Replicated at 10 repeats** (run on the M4, identical library versions):
logistic 0.662 ± 0.008, random forest 0.652 ± 0.005, hist GB 0.644 ± 0.008,
MLP 0.617 ± 0.028. The same four differences clear noise and the same four do
not, so the conclusion does not depend on the repeat count.

## F9 — The pooled-per-repeat AUC is four times more stable than a single split

Single school-split, 12 draws (F7): logistic 0.657 ± **0.037**.
5 folds × 5 repeats, pooled per repeat (F8): logistic 0.662 ± **0.010**.

Same model, same data, same split strategy — the mean barely moves (+0.005,
well inside either spread), but the spread shrinks by a factor of 3.7, because
every student is scored once per repeat instead of a quarter of them being
scored once. This is why `ecri cv` replaced `ecri baseline` as the way results
are quoted, and why the differences in F8 can be resolved at all.

## F10 — Imputation strategy makes no measurable difference

Four strategies under logistic regression, plus native-NaN under gradient
boosting, all on identical folds (5 × 5, school-disjoint). `make impute`.

| Strategy | AUC under logistic | AUC under hist GB |
|---|---|---|
| median + indicator | 0.662 ± 0.010 | 0.647 ± 0.005 |
| median only (no indicator) | 0.663 ± 0.011 | 0.647 ± 0.005 |
| mean + indicator | 0.662 ± 0.010 | 0.646 ± 0.006 |
| iterative (model-based) + indicator | 0.660 ± 0.010 | 0.642 ± 0.004 |
| no imputation, raw NaN | — | 0.644 ± 0.006 |

**Zero of seven paired differences clear noise.** The widest is 0.004 with an
interval of [−0.005, +0.013].

This is the answer F5 predicted and the ticket existed to establish rather than
assume: on the 946 labelled rows the gaps are small (worst column 24.8%, most
under 6%), so there is little for an imputer to get wrong. The expensive option
— `IterativeImputer`, which fits a model per column — is the only one that is
ever numerically worst, and even that is inside the noise.

Two things follow:

- **Keep median + indicator.** It is the cheapest, it is never worse, and the
  indicator columns cost nothing. Dropping the indicators is also defensible
  (`median_only` is nominally top under logistic by 0.001) but the difference
  is not a result, and the indicators carry the block-missingness structure
  that a later fairness or slice analysis may want.
- **Do not spend phase-2 time on imputation.** The ticket is closed with a
  measured negative, which is worth more than an untested assumption in either
  direction.

## F11 — At any plausible cost ratio the model says "screen nearly everyone"

Out-of-fold logistic scores, threshold chosen to minimise
`cost_ratio × (missed at-risk) + (false flags)`. `make calibrate`.

| Cost ratio | Threshold | Students flagged | Recall (at-risk) | Precision |
|---|---|---|---|---|
| 1:1 | 0.506 | 44.2% | 57.9% | 62.0% |
| 2:1 | 0.750 | 85.3% | 91.3% | 50.6% |
| 3:1 | 0.827 | **94.8%** | 97.8% | 48.7% |
| 5:1 | 0.989 | 100% | 100% | 47.3% |
| 10:1 | 0.989 | 100% | 100% | 47.3% |

**This is the most consequential finding for the product, and it is a negative
one.** At 3:1 — a mild assumption, since missing a struggling reader plainly
costs more than a second look at one who is fine — the cost-optimal rule flags
19 students in 20. At 5:1 it flags everyone. The model's recommendation
degenerates into blanket screening.

The arithmetic behind it is not subtle. 47.3% of this population falls short of
adequate growth, and the ranking is only 0.66 AUC. When the adverse outcome is
that common and the signal that weak, the cheapest way to avoid the expensive
error is to flag almost everybody, and a weak ranker cannot beat that.

What follows:

1. **Do not ship a 3:1 threshold and call it targeting.** It is blanket
   screening with extra steps, and presenting it as a model decision would
   misrepresent what the model contributes.
2. **The honest operating point is near 1:1** (threshold ≈ 0.51, flagging 44%,
   catching 58% of at-risk students at 62% precision). That is a real, if
   modest, improvement over flagging at random, and it is the point to put in
   the app and the model card.
3. **The useful output is the ranking, not the flag.** With a fixed support
   budget — "we can give Tier-2 help to 30 more children this term" — the model
   earns its keep by ordering the queue. Framing it as a yes/no screen is what
   makes it look useless.
4. **The cost ratio is an assumption and must be stated wherever a threshold
   appears.** It is a single constant in `calibrate.DEFAULT_COST_RATIO`, and
   moving it from 1 to 3 more than doubles the number of children flagged.

Calibration itself is not broken: raw Brier 0.2301 against 0.2278 for Platt and
0.2232 for isotonic, mean absolute reliability gap 0.052 raw against 0.036
Platt. Those are fitted and scored on the same out-of-fold predictions, so they
are an upper bound — isotonic's 0.0000 gap is in-sample overfit, not a result —
but the gap between raw and calibrated is small enough that the raw scores are
usable as probabilities with a Platt step on top.

## F12 — Five or six features carry everything; the rest is noise

Permutation importance on held-out folds (5 folds × 3 repeats, school-disjoint),
measured as the drop in ROC-AUC when one raw column is shuffled. `make importance`.

| Feature | logistic | random forest |
|---|---|---|
| `RMwidRS` (baseline Word ID) | **0.1220 ± 0.0421** | **0.0856 ± 0.0289** |
| `rcmistot` (classroom management rating) | 0.0296 ± 0.0339 | 0.0028 |
| `RMwdaRS` (baseline Word Attack) | 0.0291 ± 0.0201 | 0.0135 ± 0.0074 |
| `Tier2MinRd` (Tier-2 reading minutes) | 0.0190 ± 0.0190 | 0.0044 ± 0.0075 |
| `NWFwrc` (nonsense-word recoding) | 0.0144 ± 0.0109 | 0.0207 ± 0.0155 |
| `SAsrS` (Stanford sentence reading) | 0.0117 ± 0.0105 | 0.0019 |
| `SpEd` | 0.0097 ± 0.0083 | 0.0061 ± 0.0036 |
| `Age1b` | 0.0068 ± 0.0071 | 0.0074 ± 0.0060 |

Above noise (mean > SD): **6 of 22** under logistic (`RMwidRS`, `RMwdaRS`,
`NWFwrc`, `SAsrS`, `SpEd`, `Age1b`), **5 of 22** under random forest
(`RMwidRS`, `NWFwrc`, `RMwdaRS`, `Age1b`, `SpEd`).

Four things:

**Baseline Word ID is not merely first, it is an order of magnitude ahead** —
0.122 against 0.030 for the runner-up. Removing it costs more AUC than removing
everything else put together. This is coherent with F1: the label is the gain in
that exact measure, so its starting value sets the headroom.

**Two unrelated model families agree.** Spearman ρ = 0.608 across all 22
features, and their above-noise shortlists overlap on four of five
(`RMwidRS`, `RMwdaRS`, `NWFwrc`, `SpEd`). Agreement between a linear model and a
500-tree ensemble is the strongest evidence available here that these rankings
reflect the data rather than one model's inductive bias.

**`rcmistot` is the trap in this table.** It ranks second under logistic — and
its SD (0.0339) exceeds its mean (0.0296), while the random forest puts it at
0.0028, essentially nothing. Anyone reading the sorted column without the
spread would report "classroom management quality is the second most important
predictor of reading growth", which the data does not support. The figure draws
error bars for exactly this reason.

**The classroom and teacher block contributes almost nothing measurable.**
`grp_rate`, `gnrl_fid`, `TKPctCrct`, `TYearsTeaching`, `Tier2_N` and `Tx` are
all inside noise under both models. Note that this sits in tension with F7,
where the full feature set beat the baseline-score-only model by +0.073 on Word
ID — so the extra features do contribute in aggregate, just not in a way that
survives attribution to any individual column. Both can be true: many weak,
partly redundant signals.

## F13 — School context: a borderline gain from poverty, nothing from race

`MetaData.csv` gives each school's free/reduced-price-lunch rate and racial
composition. Three feature sets under logistic regression on identical folds.
`make school`.

| Feature set | 5 repeats | **10 repeats** |
|---|---|---|
| student + classroom + teacher only | 0.662 ± 0.010 | 0.662 ± 0.008 |
| + school free-lunch rate | 0.673 ± 0.008 | **0.672 ± 0.007** |
| + free-lunch + racial composition | 0.664 ± 0.013 | 0.664 ± 0.010 |

| Paired difference | 5 repeats | **10 repeats** |
|---|---|---|
| + free-lunch | +0.012 [+0.001, +0.023] — clears | **+0.009 [−0.002, +0.021] — within noise** |
| + composition | +0.003 [−0.015, +0.022] — within | +0.003 [−0.015, +0.022] — within |

**The free-lunch result does not survive replication, and that is the reason
it is recorded here rather than shipped.** At 5 repeats the interval cleared
zero — and cleared it in 9 of 10 bootstrap seeds, which looked solid. At 10
repeats the interval crosses zero. The point estimate barely moved (+0.012 →
+0.009); what moved is the lower bound, which was sitting within 0.001 of zero
all along. A result that flips with the repeat count is not a result.

So:

- **Free-lunch rate stays out of the default feature set.** The join exists, is
  tested, and is one flag away (`join_school_meta`). It is a candidate worth
  re-testing once the paper replication (T33) fixes the evaluation protocol,
  not a finding to report.
- **Racial composition stays out permanently.** It adds nothing — on top of
  free-lunch rate it *costs* 0.008 AUC — and a model that routes children to
  reading support must not learn to allocate it by a school's racial make-up.
  Here the ethical answer and the empirical answer coincide, which makes the
  decision easy; `schema.SCHOOL_COMPOSITION` is kept separate and opt-in so
  that it stays easy.

The general lesson for the rest of phase 2: **re-run anything borderline at a
higher repeat count before writing it down.** A lower bound within 0.001 of
zero is a coin flip, whatever the bootstrap says.
