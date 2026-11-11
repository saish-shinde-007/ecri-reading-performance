# Plan

38 tickets. **16 shipped (phase 1, ~40%)**, 22 remaining (phase 2, ~60%).
One ticket is one commit. Every ticket below carries a definition of done, so
"finished" is a check rather than an opinion.

Tickets are numbered by commit, not by plan order — T14 was folded into T13
(the Markdown renderer belonged with the report that needed it).

---

## Phase 1 — shipped

| # | Ticket | Done |
|---|---|---|
| T01 | Scaffold: packaging, licence, make targets, data layout | ✅ |
| T02 | Docs: proposal, file inventory, data licence | ✅ |
| T03 | Paths and typed config | ✅ |
| T04 | Data contract for the 30 published columns | ✅ |
| T05 | **Derive the label definition; verify it reproduces exactly** | ✅ |
| T06 | Synthetic generator matching the real structure | ✅ |
| T07 | Loader with a validating contract check | ✅ |
| T08 | Cleaning and modelling-frame assembly | ✅ |
| T09 | School-disjoint splits with an enforcing leakage guard | ✅ |
| T10 | Feature pipeline with missingness indicators | ✅ |
| T11 | Classification metrics, at-risk-recall focused | ✅ |
| T12 | Baselines, and the bar they set | ✅ |
| T13 | EDA + Markdown rendering | ✅ |
| T15 | CLI: check, synth, eda, baseline | ✅ |
| T16 | Test suite (124 tests, 96%), CI, three bug fixes | ✅ |
| T17 | README and this plan | ✅ |

**What phase 1 bought:** a pipeline that runs end to end on real or synthetic
data, a label definition established by proof rather than assumption, a leakage
guard that fails runs instead of warning, and a measured bar — logistic
regression at **0.657 ± 0.037** school-split AUC on Word Identification — that
every phase-2 model has to clear.

---

## Phase 2 — remaining

Ordered by dependency, not by appeal. **T18 comes first and blocks the rest**:
without it, model comparisons are noise.

### Stage A — make comparisons trustworthy

#### T18 · Repeated cross-validation harness
**Blocks T19–T25.** Single splits have an AUC SD near 0.037 (F7), so two models
within 0.07 of each other are indistinguishable on one split. Build a harness
that runs repeated stratified-group CV and returns mean ± SD per model.
*Done when:* `ecri cv --model X --repeats N` reports mean ± SD; the baselines
rerun through it and reproduce F7 within 0.01; single-split numbers no longer
appear anywhere in a report.

#### T19 · Paired comparison with uncertainty
Paired bootstrap or corrected resampled t-test over the same folds, so "model A
beats model B" is a claim with an interval attached.
*Done when:* the comparison table carries a difference, a CI, and a plain
statement of whether the difference clears noise.

### Stage B — models

#### T20 · Tree ensembles
RandomForest and HistGradientBoosting (stdlib sklearn, no new dependency).
HistGradientBoosting takes NaN natively, which is a genuine alternative to
imputation given F5.
*Done when:* both run through T18 and land in the comparison table against the
0.657 bar; native-NaN handling is compared to the imputing pipeline.

#### T21 · XGBoost / LightGBM
Behind the existing `boost` extra, so the core install stays small.
*Done when:* both are optional imports that skip cleanly when absent, CI
installs the extra in one matrix leg, and results join the table.

#### T22 · MLP
`MLPClassifier` with a small architecture search. The paper's method is an MLP,
so this is the honest comparison point.
*Done when:* tuned through T18, in the table, with the search space recorded.

#### T23 · Masked-input pre-training (the paper's MaskMLP)
Reproduce the paper's contribution: pre-train the MLP on randomly masked inputs
with a cosine embedding loss between masked and unmasked embeddings, then
fine-tune. Given F5's block missingness, masking whole blocks rather than
individual cells is the variant worth testing.
*Done when:* implemented, compared to the plain MLP through T18, and the result
stated plainly whether or not it beats the simpler model.

#### T24 · Regression variant
The raw gain is available from `ECRIRawData.csv`, so the thresholded label
throws away information. Predict the gain directly and threshold afterwards.
*Done when:* RMSE/MAE/R² reported, and the thresholded regressor compared to
direct classification on the same folds.

#### T25 · Imputation comparison
Median vs iterative vs native-NaN vs indicator-only. F5 notes the gaps are
small on labelled rows, so the expected effect is small — the ticket exists to
establish that rather than assume it.
*Done when:* the comparison runs through T18 and the README states the finding
even if it is "no material difference".

### Stage C — interpretation

#### T26 · Permutation importance
Model-agnostic, computed on held-out folds.
*Done when:* a ranked table plus a figure, for the best model and for logistic.

#### T27 · SHAP explanations
Behind the `explain` extra.
*Done when:* global summary plot plus per-student explanations, and the
agreement (or not) with T26 is stated.

#### T28 · Calibration and threshold selection
A risk score used for allocating intervention needs calibrated probabilities
and a defensible cut point, not 0.5. Reliability curves, Platt/isotonic, and a
threshold chosen by cost ratio.
*Done when:* calibration curves are in the report and the chosen threshold is
justified by a stated cost assumption.

#### T29 · Fairness slice analysis
Per-slice metrics for `LEP`, `SpEd`, `Gender`, `Tx`, and school free-lunch rate
from `MetaData.csv`. A tool that routes support to children must not work
noticeably worse for one group.
*Done when:* a slice table with per-slice AUC and recall-at-risk, and any gap
larger than the measurement noise is called out in the model card (T35).

### Stage D — data depth

#### T30 · Join school-level context
`MetaData.csv` carries free/reduced-price lunch and composition for all 44
schools and is currently unused.
*Done when:* joined on `SchlID`, the leakage guard still passes, and the effect
on the comparison table is reported.

#### T31 · Richer baseline features from the longitudinal file
`ECRIRawData.csv` holds `SAwsS` and `SArcS`, which the analysis table drops.
Pull **t=0 rows only** and extend the feature set.
*Done when:* the extra baseline measures are joined, `assert_no_leakage` passes,
and a test proves a t≥0.5 row cannot enter the join.

#### T32 · Replicate the paper's reported numbers
Compare against the published School-split and student-split results.
*Done when:* the comparison is tabulated with any gap explained, or the
discrepancy is documented as unexplained.

### Stage E — delivery

#### T33 · Model persistence
joblib artefact plus a JSON sidecar recording data source, git SHA, config,
feature names and CV scores. A saved model with no provenance is not reusable.
*Done when:* save/load round-trips in a test and a loaded model reproduces its
recorded score.

#### T34 · Inference module
`predict(records) -> risk scores` against a stable input schema, validating
inputs and refusing unknown or leaking columns.
*Done when:* tested on records the model never saw, including ones with missing
fields.

#### T35 · Model card and ethics statement
What the model is for, who it was fit on (Tier-2 at-risk readers in 44 schools,
946 labelled students), measured performance with uncertainty, slice results
from T29, and explicit limits: a 0.66-AUC screening aid is a prompt to look
more closely, never a placement decision. The proposal's own framing —
decision support, not replacement — belongs here as a constraint.
*Done when:* `docs/model_card.md` exists and the Streamlit app links to it.

#### T36 · Streamlit app
The proposal's "Early Reading Risk Predictor": enter baseline assessment and
classroom values, get a calibrated risk estimate with its drivers.
*Done when:* runs from the `app` extra, loads a T33 artefact, shows the
uncertainty and the model-card limits on screen rather than a bare number.

#### T37 · Final report
Generated from the committed artefacts so it cannot drift from the code.
*Done when:* `make report` writes a report whose every number traces to a
reproducible run.

#### T38 · Reproducibility pass
Lockfile, seed audit, `make all` from a clean clone, documented runtimes.
*Done when:* a fresh clone reproduces the comparison table to 3 decimal places.

---

## Deferred, with reasons

- **Class imbalance handling** (the proposal's challenge #2). The classes are
  52.7% / 57.6% positive (F4) — near balanced. It becomes relevant only if the
  target is redefined at a stricter growth threshold; until then it is work
  with nothing to fix.
- **Deep tabular architectures** (TabNet, FT-Transformer). 946 labelled rows and
  22 features. The data does not support them, and T23 already covers the
  paper's own neural contribution.
