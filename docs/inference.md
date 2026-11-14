# The model contract

Everything the model card (T36, [#9](https://github.com/saish-shinde-007/ecri-reading-performance/issues/9))
and the Streamlit app (T37, [#10](https://github.com/saish-shinde-007/ecri-reading-performance/issues/10))
need from the model, in one place. Built in T27 (`ecri/persist.py`) and T28
(`ecri/inference.py`).

**Short version:** run `make model`, load the result with
`persist.load_bundle()`, call `inference.rank(records, bundle, budget=k)`, show
the table it returns — including its `notes` column — and read the model card's
numbers from `model.json`.

---

## 1. Get a model

```bash
make data     # once: fetch the ECRI CSVs into data/raw (CC BY-NC-SA 4.0)
make model    # train, calibrate, save and verify models/logistic-RMwidRS_target/
```

That takes about ten seconds on a laptop. The bundle is git-ignored; everyone
trains their own from the same data and gets the same model, which
`ecri model verify --cv` can prove (§6).

Without the real data, `make model` still works: it trains on synthetic data
and says so on every line it prints. That is enough to build and test the app,
and the sidecar records `"data": {"source": "synthetic"}` — the app must show a
banner when it sees that (§4).

## 2. Load it

```python
from ecri import persist

bundle = persist.load_bundle("models/logistic-RMwidRS_target")
bundle.features        # the 22 inputs, in training order
bundle.metadata        # the parsed model.json (§5)
```

`load_bundle` refuses a bundle that does not line up, and says why:

| Check | Raises | What it catches |
|---|---|---|
| format version | `IntegrityError` | a bundle from a newer or older format |
| `model.joblib` sha256 vs sidecar | `IntegrityError` | a corrupted, edited or swapped model file |
| pipeline features vs sidecar | `IntegrityError` | a sidecar paired with the wrong model |
| `expected_features=` (optional) | `IntegrityError` | the caller expecting a different input set |
| stored probe predictions | `IntegrityError` | a library upgrade that changed the model's outputs |
| no `model.json` | `FileNotFoundError` | nothing trained yet — the message says to run `make model` |

If the probe check fails after a `pip` upgrade, retrain with `make model`.

> **A joblib file is a pickle: loading one runs code from it.** Load only
> bundles you trained or trust. The sha256 detects a changed file; it does not
> prove who made it. To inspect a bundle without loading it, use
> `persist.read_metadata(path)` or `ecri model show`, which read only the JSON.

## 3. Rank students

```python
from ecri import inference

ranking = inference.rank(records, bundle, budget=30)

ranking.table       # the queue, most at risk first (below)
ranking.warnings    # tuple of plain-English notes for the whole batch
ranking.summary()   # one line, e.g. "ranked 120 student(s) with logistic → ..."
ranking.data_source # "real" or "synthetic" — what the MODEL was trained on
```

### Input

`records` can be a DataFrame, one dict, or a list of dicts. Each record holds:

- **an id column** — `StuID` by default; change it with `id_column=`. Without
  one, rows are numbered from 1 under `row`.
- **any of the 22 model inputs** — `bundle.features`. All are numbers: scores,
  or codes such as `Gender` 0/1 and `Tier` 1–3. Numbers stored as text (`"12"`)
  are fine.
- **blanks are allowed.** A blank is filled with the training median and listed
  in `missing_fields`. The deployed model has no missingness flags, so a blank
  cannot push a student up or down the queue on its own (F14).
- `SchlID` and `TeachID` may be present; they are passed over, never used.

Refused, each with a message naming the problem:

| Input | Raises | Why |
|---|---|---|
| a label column (`RMwidRS_target`) or a post-baseline score (`RMwidRS_t1.0`) | `LeakageError` | the outcome is already known, or future data was merged in. Refused even with `ignore_unknown=True` |
| a column the model does not use | `InputError` | pass `ignore_unknown=True` to drop it instead; the drop is listed in `warnings` |
| text that is not a number (`"twelve"`) | `InputError` | names the column |
| a value outside its plausible range | `InputError` | names the student and value — `Age1b=790` is a slipped decimal, not an old child. Ranges are in `clean.PLAUSIBLE_RANGES` |
| no records | `InputError` | |

### Output: `ranking.table`

One row per student, sorted most at risk first. Equal scores keep input order,
so the same input always gives the same queue.

| Column | Type | Meaning |
|---|---|---|
| `rank` | int | 1 = most at risk |
| `StuID` (or `row`) | as given | the id column |
| `p_adequate_growth` | 0–1 | calibrated probability the student makes adequate growth (gains ≥ 27 on Woodcock Word ID, F1) |
| `risk` | 0–1 | `1 − p_adequate_growth` |
| `risk_percentile` | 0–100 | share of the 946 evaluation students less at risk. 90 = riskier than 90% of them |
| `features_present` | int | how many of the 22 inputs were given |
| `missing_fields` | str | comma-separated names of the blanks |
| `notes` | str | why this score deserves less weight; empty when there is no caveat |

With `budget=k`, only the top `k` rows come back. A budget larger than the
input returns everyone and adds a warning.

**There is no yes/no column, by design.** If missing a struggling reader counts
even twice as much as a false alarm, the cost-optimal threshold flags 91% of
these students (F11). A flag would mean "screen almost everyone"; a ranked
queue under a budget means "help these first".

### Scope: who the model can rank (F15)

`notes` marks two things:

- **`Tier 1` / `Tier 3`** — the model was fit 98.4% on Tier-2 readers.
- **`no baseline RMwidRS`** — the input the score leans on most is missing.

On the real data these notes cover 1.6% of the Tier-2 students the model was
built for, and **100% of everyone else**: the Woodcock assessment was given only
to the Tier-2 subsample. The model ranks Tier-2 readers who have a baseline
Woodcock score. It is not a whole-school screener, and the app must not present
it as one.

## 4. Building the app (T37)

A sketch of the intended wiring, not the app itself:

```python
import pandas as pd
import streamlit as st
from ecri import inference, persist

@st.cache_resource
def load():
    return persist.load_bundle("models/logistic-RMwidRS_target")

bundle = load()
if bundle.metadata["data"]["source"] != "real":
    st.warning("This model was trained on synthetic data. Scores mean nothing about real students.")

upload = st.file_uploader("Students (CSV)")
budget = st.slider("How many students can you support this term?", 1, 100, 20)
if upload:
    try:
        ranking = inference.rank(pd.read_csv(upload), bundle, budget=budget, ignore_unknown=True)
    except (inference.InputError, inference.LeakageError) as err:
        st.error(str(err))
    else:
        for note in ranking.warnings:
            st.info(note)
        st.dataframe(ranking.table)   # keep `notes` visible
```

The checklist from #10, against this contract:

- [ ] Load once and cache (`st.cache_resource`).
- [ ] Budget slider → `rank(..., budget=k)`.
- [ ] Show `p_adequate_growth`, `risk_percentile`, `missing_fields` **and
      `notes`** for every student.
- [ ] Banner when `metadata["data"]["source"] != "real"`.
- [ ] Never show a bare yes/no. If a threshold appears at all, show
      `operating_points.recommended_cost_ratio` beside it.
- [ ] Link to `docs/model_card.md`.
- [ ] For a quick input file: `ecri synth --features-only` writes one, drawn
      from in-scope students (a baseline Woodcock score) so the queue looks the
      way it should in real use.

## 5. The metadata sidecar: `model.json`

Everything the model card (T36) needs is here, so nothing in the card is typed
by hand. From the real bundle, abridged:

```json
{
  "format_version": 1,
  "model": {
    "name": "logistic",
    "description": "L2 logistic regression on the full feature set",
    "preprocessor": {"numeric_impute": "median", "add_indicator": false, "scale": true, "categorical": "onehot"},
    "preprocessor_note": "missingness flags off for deployment (F14)"
  },
  "target": "RMwidRS_target",
  "features": ["Gender", "LEP", "SpEd", "Tier", "… 18 more"],
  "code": {"ecri": "0.1.0", "git_sha": "1bb3221ebe53…", "git_dirty": false},
  "environment": {"python": "3.12.3", "sklearn": "1.9.1", "numpy": "2.5.4", "pandas": "3.1.1"},
  "data": {"source": "real", "file": "ECRIDataSet.csv", "rows": 946, "positive_rate": 0.5275, "schools": 44},
  "evaluation": {"strategy": "school", "n_folds": 5, "repeats": 5, "seed": 20260101},
  "cv": {"auc_mean": 0.662506, "auc_sd": 0.010692, "ap_mean": 0.661863,
         "recall_at_risk_mean": 0.547204, "brier_mean": 0.230756},
  "calibration": {"method": "platt", "in_sample": true, "brier_raw": 0.229368, "brier_calibrated": 0.227882},
  "operating_points": {
    "recommended_cost_ratio": 1.0,
    "recommended": {"threshold": 0.5017, "flagged_rate": 0.4144, "recall_at_risk": 0.5503, "precision_at_risk": 0.6276},
    "sweep": [{"cost_ratio": 1.0, "flagged_rate": 0.414}, {"cost_ratio": 2.0, "flagged_rate": 0.91}, "…"]
  },
  "population": {"tier_share": {"2": 0.9841, "1": 0.0095, "3": 0.0063},
                 "training_tiers": [2.0], "key_feature": "RMwidRS", "key_feature_missing_rate": 0.0}
}
```

| Field | Use it for |
|---|---|
| `cv.auc_mean`, `cv.auc_sd` | headline performance, always with its SD and `evaluation.strategy` |
| `cv.recall_at_risk_mean` | share of at-risk students caught at the 0.5 cut |
| `calibration.*` | whether probabilities can be read as probabilities. `in_sample: true` means the Brier is optimistic |
| `operating_points.sweep` | how many students each cost assumption would flag — the F11 table |
| `population.*` | the scope statement (F15) |
| `data.*`, `code.*`, `environment.*` | provenance: what data, which commit, which library versions |

**Stability:** fields may be added. Renaming or removing one bumps
`format_version`, and old bundles then refuse to load with a clear message
rather than half-working.

## 6. Command line

```bash
ecri model train [--model logistic] [--repeats 5] [--out DIR]
ecri model show [DIR] [--json]       # sidecar only; never unpickles
ecri model verify [DIR] [--cv]       # integrity; --cv re-derives the CV AUC to 3 dp
ecri rank STUDENTS.csv --budget 20 [--out ranked.csv] [--ignore-unknown]
make rank-demo                       # model + synthetic input + ranking, end to end
```

`rank` and `model verify` exit 2 with a one-line message on input problems.
`verify --cv` exits 1 if the score does not reproduce.

## 7. Tests

`tests/conftest.py` provides three session fixtures, synthetic throughout, so
app tests need no real data and no training of their own:

- `bundle` — a trained `ModelBundle`
- `bundle_dir` — the same bundle saved to a temp directory
- `unseen_records` — students the bundle never saw, outcomes removed

```python
def test_app_shows_the_top_k(bundle, unseen_records):
    ranking = inference.rank(unseen_records, bundle, budget=5)
    assert len(ranking.table) == 5
```
