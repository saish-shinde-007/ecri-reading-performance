# Plan

39 tickets. **70% shipped**: phase 1, phase 2a, and all of our phase-2b
tickets (T27, T28, T39). What remains is Sarvesh's 30% (T29–T38, issues
#2–#11). The final 40% was split 30% Sarvesh / 10% us, weighted by effort
rather than ticket count.

| Slice | Tickets | Share | Status |
|---|---|---|---|
| Phase 1 — foundation | T01–T18 | 40% | ✅ shipped |
| Phase 2a — evaluation and models | T19–T26 | 20% | ✅ shipped (PR) |
| Phase 2b — **Sarvesh** | T29–T38 (issues #2–#11) | 30% | open, unblocked |
| Phase 2b — **us** | T27, T28, T39 | 10% | ✅ shipped (tag `v0.9.0`) |

Each ticket is one commit (or one small PR) with a definition of done, so
"finished" is a check rather than an opinion.

---

## Shipped

### Phase 1 — foundation (40%)

| # | Ticket |
|---|---|
| T01 | Scaffold: packaging, licence, make targets, data layout |
| T02 | Docs: proposal, file inventory, data licence |
| T03 | Paths and typed config |
| T04 | Data contract for the 30 published columns |
| T05 | **Derive the label definition; verify it reproduces exactly** |
| T06 | Synthetic generator matching the real structure |
| T07 | Loader with a validating contract check |
| T08 | Cleaning and modelling-frame assembly |
| T09 | School-disjoint splits with an enforcing leakage guard |
| T10 | Feature pipeline with missingness indicators |
| T11 | Classification metrics, at-risk-recall focused |
| T12 | Baselines, and the bar they set |
| T13 | EDA + Markdown rendering (T14 folded in) |
| T15 | CLI: check, synth, eda, baseline |
| T16 | Test suite, CI, three bug fixes |
| T17 | README and plan |
| T18 | CI fixes: venv per matrix leg, interpreter assertion, errexit |

### Phase 2a — evaluation and models (20%)

| # | Ticket | Result |
|---|---|---|
| T19 | Model registry + parameterised preprocessor | 9 models, 2 optional |
| T20 | Repeated-CV harness + paired bootstrap | spread 0.037 → 0.010 (F9) |
| T21 | `ecri cv` — 9-model comparison | **nothing beats logistic** (F8) |
| T22 | Imputation comparison | no measurable difference (F10) |
| T23 | Permutation importance | 5–6 of 22 features carry it (F12) |
| T24 | Calibration + threshold selection | **3:1 cost ⇒ flag 95%** (F11) |
| T25 | School-level context join | free-lunch unconfirmed; race out (F13) |
| T26 | Tests for phase 2a, CI, make targets | 184 tests, 94% |

### Phase 2b — our tickets (10%)

| # | Ticket | Result |
|---|---|---|
| T27 | Model persistence + provenance sidecar | bundle reproduces its CV AUC to 6 digits; 5 integrity checks |
| T28 | `rank(records, bundle, budget)` | the contract in [docs/inference.md](docs/inference.md) |
| T39 | Reproducibility gate | lockfile, pinned data, seed audit, `make reproduce`, generated README — [docs/reproducibility.md](docs/reproducibility.md) |

Building T28 found two problems no metric had shown: a blank age cell moved a
student 50 percentile points (**F14**, fixed — the deployed model has no
missingness flags, at no cost in AUC), and the model's key input is missing for
every student outside the Tier-2 subsample (**F15** — the app ranks Tier-2
readers, it does not screen a whole school).

**What phase 2a changed.** Three results reshape the remaining work:

1. **Logistic regression is the model** (F8). Every tree ensemble and the MLP
   sit within noise of it at 10 repeats. Remaining modelling tickets are
   expected to confirm this, not overturn it — and are scoped accordingly.
2. **The product is a ranking, not a flag** (F11). At any cost ratio above
   ~2:1 the cost-optimal threshold flags nearly every student. The app and the
   inference contract present a ranked queue under a support budget.
3. **Borderline results get re-run before they are written down** (F13). The
   free-lunch gain cleared noise at 5 repeats and failed at 10.

---

## Phase 2b — open (40%)

Ordered by dependency. **T27 and T28 are done**, so nothing in Sarvesh's list
is blocked on us any more. The contract they expose is
[docs/inference.md](docs/inference.md).

```
T27 persistence ──► T28 inference ──► T37 Streamlit app ──┐
      │                                                     │
      └──────────► T36 model card ◄── T34 fairness          ├──► T38 report ──► T39 release
                                                            │
T29 boost · T30 regression · T31 t0 features · T32 MaskMLP ─┤
T33 SHAP · T35 paper replication ───────────────────────────┘
```

### Us — 10%

#### T27 · Model persistence — 2% · ✅ shipped
joblib artefact plus a JSON sidecar: data source, git SHA, config, feature
names, CV scores, chosen threshold and the cost ratio it assumes. A saved model
with no provenance is not reusable.
*Done when:* save/load round-trips in a test; a loaded model reproduces its
recorded out-of-fold AUC to 3 dp; the sidecar refuses to load against a
different feature list.

#### T28 · Inference module — 4% · ✅ shipped
`rank(records, budget=None)` against a stable input schema. Validates inputs,
refuses unknown or leaking columns through the existing guard, and returns a
**ranked queue with percentile and calibrated probability** — not a yes/no
flag (F11). With a `budget`, returns the top-k.
*Done when:* tested on records the model never saw, including missing fields;
a record with a post-baseline column is rejected; a budget of k returns exactly
k students in score order.

#### T39 · Reproducibility pass and release — 4% · ✅ shipped

Shipped as `v0.9.0`. The `v1.0.0` tag waits for T38, because a release
certifies the numbers the final report quotes; cutting it is now
`make release-check` plus a tag ([RELEASING.md](RELEASING.md)).

Lockfile, seed audit, `make all` from a clean clone on both the laptop and the
M4, documented runtimes, tagged `v1.0`.
*Done when:* a fresh clone reproduces the F8 table to 3 dp; every report in
`reports/` regenerates from `make all`; the README results table is generated,
not hand-typed.

### Sarvesh — 30%

#### T29 · XGBoost and LightGBM — 1.5%
Install the `boost` extra and run both through `ecri cv`. The specs already
exist in `zoo.py`. **Expectation (F8):** they are the same family as the two
gradient-boosting entries already measured below logistic, so a tie is the
likely outcome — and a fine one to report.
*Done when:* both appear in the comparison table at 10 repeats; one CI leg
installs the extra.

#### T30 · Regression on raw gain — 3%
The label throws information away: it is `gain ≥ 27` (F1). Predict the gain
itself from `ECRIRawData.csv` (t=1 minus t=0 as the **target only**), then
threshold the prediction.
*Done when:* RMSE/MAE/R² reported; the thresholded regressor is compared to
direct classification on the same folds with the paired bootstrap; a test
proves no t≥0.5 value reaches the features.

#### T31 · Richer baseline features from the raw file — 2.5%
`ECRIRawData.csv` holds `SAwsS` (word study) and `SArcS` (comprehension), which
the analysis table drops. Join **t=0 rows only**.
*Done when:* the two measures are joined; `assert_no_leakage` passes; a test
proves a t≥0.5 row cannot enter the join; the effect is measured at 10 repeats.

#### T32 · Masked-input pre-training (the paper's MaskMLP) — 5%
Reproduce the paper's method: pre-train an MLP on randomly masked inputs with a
cosine-embedding loss between masked and unmasked embeddings, then fine-tune.
Given F5's block missingness, also try masking whole blocks (the teacher
survey, the assessment session) rather than independent cells.
*Done when:* implemented, compared against the plain MLP **and logistic** at 10
repeats, and the result stated plainly whether or not it wins. The plain MLP
already sits within noise of logistic with 3.5× its variance (F8).

#### T33 · SHAP explanations — 2.5%
Behind the `explain` extra.
*Done when:* global summary plot plus per-student explanations for logistic;
agreement with T23's permutation ranking stated (Spearman ρ), and any
disagreement explained.

#### T34 · Fairness slice analysis — 3%
Per-slice AUC and recall-at-risk for `LEP`, `SpEd`, `Gender`, `Tx`, and school
free-lunch tercile. Use `evaluate.oof_frame()`, which already carries these
columns. A tool that routes support to children must not work noticeably worse
for one group.
*Done when:* a slice table with bootstrap intervals per slice; any gap larger
than the noise is called out in the model card.

#### T35 · Replicate the paper's numbers — 4%
Compare against the published School-split and student-split results, under
the paper's own protocol. Re-test the free-lunch feature (F13) under that
protocol while there.
*Done when:* a side-by-side table with every gap explained, or documented as
unexplained.

#### T36 · Model card and ethics statement — 1.5%
Intended use, population (946 Tier-2 at-risk readers, 44 schools), performance
with intervals, slice results from T34, and the limits: a 0.66-AUC ranker is a
prompt to look more closely, never a placement decision. Must state the cost
ratio behind any threshold (F11).
*Done when:* `docs/model_card.md` exists, reads its numbers from the T27
sidecar, and the app links to it.

#### T37 · Streamlit app — 5%
The proposal's "Early Reading Risk Predictor", built on T28. **Frame it as a
ranked queue under a support budget** — "you can help N more children this
term; here they are, in order" — not a yes/no screen (F11). Show the
calibrated probability, the top drivers, the uncertainty, and the model-card
limits on screen.
*Done when:* runs from the `app` extra; loads a T27 artefact; has a budget
slider; never displays a bare flag without the cost ratio behind it.

#### T38 · Final report — 2%
Generated from the committed artefacts so it cannot drift from the code.
*Done when:* `make report` writes it, and every number traces to a reproducible
run.

---

## Deferred, with reasons

- **Class-imbalance handling.** 52.7% / 57.6% positive (F4) — nothing to fix
  unless the target is redefined at a stricter threshold.
- **Hyper-parameter tuning.** F8 puts every flexible model within noise of
  logistic; tuning them is optimising inside the noise band.
- **Deep tabular architectures** (TabNet, FT-Transformer). 946 labelled rows;
  T32 already covers the paper's own neural contribution.

## Compute

Heavy runs (10-repeat CV, permutation importance, MaskMLP training) can go to
a remote box rather than a laptop. Set the host once and prefix the command:

```bash
export ECRI_REMOTE=user@compute-host
ssh "$ECRI_REMOTE" 'cd ~/ecri && .venv/bin/python -m ecri.cli cv --repeats 10'
```

Put a job throttle in front of it if the box is shared, so parallel runs queue
instead of overloading it. Pin identical library versions on both machines
(`uv pip install -e ".[dev]"` from the same lockfile) and results transfer
exactly — F8's 10-repeat replication was run this way.
