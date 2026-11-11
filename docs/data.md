# Getting the data

Nothing under `data/` is committed. The ECRI files are third-party data under a
**non-commercial, share-alike licence** — see *Licence* below — so this repo
ships code and aggregate statistics only. Every test and the whole CLI run
without the real files, against the synthetic generator (`ecri.datagen`).

## Source

Everything lives in one public repo: <https://github.com/ecri-data/ECRI-Dataset>
(linked from the project page <https://ecri-data.github.io/> as both "GitHub"
and "Data"). There is no access form and no request process.

## Download

```bash
make data            # or the equivalent by hand:
curl -L -o data/raw/ECRIDataSet.csv  https://raw.githubusercontent.com/ecri-data/ECRI-Dataset/main/ECRI/ECRIDataSet.csv
curl -L -o data/raw/ECRIRawData.csv  https://raw.githubusercontent.com/ecri-data/ECRI-Dataset/main/ECRI/ECRIRawData.csv
curl -L -o data/raw/MetaData.csv     https://raw.githubusercontent.com/ecri-data/ECRI-Dataset/main/ECRI/MetaData.csv
```

`ecri.loader.load_analysis()` checks these paths and falls back to synthetic
data (with a loud warning) when they are absent.

## File inventory

Verified against the repo at commit state of 2024-08-19.

| File | Size | Rows | Cols | What it is |
|---|---|---|---|---|
| `ECRI/ECRIDataSet.csv` | 940 KB | 6,803 | 30 | **The paper's analysis table.** One row per student, the baseline time point only (`Time`=1.0, `t`=0.0), plus the two binary `*_target` labels. This is the file to model on. |
| `ECRI/ECRIRawData.csv` | 4.2 MB | 27,664 | 31 | **The longitudinal file.** 6,916 students × 4 time points (`t` ∈ {0, 0.5, 1, 2}), no label columns. Adds two assessments the analysis table drops: `SAwsS` (word study) and `SArcS` (reading comprehension). |
| `ECRI/MetaData.csv` | 5 KB | 44 | 9 | School-level context: free/reduced-price-lunch rate and racial composition per `SchoolID`. Joinable to `SchlID`. |
| `ECRI_croissant.json` | 28 KB | — | — | Croissant ML metadata record. |

Counts differ slightly from the paper's headline 6,916 students / 172 teachers:
the analysis table carries 6,803 students and 171 distinct `TeachID` values, so
113 students were dropped between the raw and analysis files.

### `Wave` is not the time index

`ECRIRawData.csv` has both a `Wave` column (values `"Wave 1"`, `"Wave 2"`) and a
`t`/`Time` pair. `Wave` is the study recruitment group — 8,528 and 19,136 rows,
not a 4-way split. The **time index is `t`** (0, 0.5, 1, 2), four rows per
student, exactly 6,916 students at each value. Do not treat `Wave` as a wave.

## The prediction timeline

The label is a *growth* outcome measured between `t=0` and `t=1` — see
`docs/findings.md`, which derives the exact rule. That makes the leakage
boundary sharp:

- **Legal features:** anything measured at `t=0`, which is everything in
  `ECRIDataSet.csv` apart from the two `*_target` columns.
- **Leakage:** any `t ≥ 0.5` value pulled in from `ECRIRawData.csv`. In
  particular `RMwidRS`/`RMwdaRS` at `t=1` reconstruct the label outright.

`ecri.split.assert_no_leakage()` enforces this against the deny-list in
`ecri.schema.LEAKING_COLUMNS` and fails the run rather than warning.

## Licence

The dataset repo is **CC BY-NC-SA 4.0**
(<http://creativecommons.org/licenses/by-nc-sa/4.0/>). GitHub's licence
detector reports `NOASSERTION` because the badge sits in the README, so the
README is the authority. Practical consequences for this project:

- **BY** — cite Shangguan et al. (2024) in any output.
- **NC** — coursework and research use is fine; commercial use is not.
- **SA** — redistributing the data, or a derivative of it, would oblige you to
  license that under CC BY-NC-SA 4.0 too. Committing the CSVs here would drag
  this MIT-licensed code into that obligation, which is the second reason
  `data/` is git-ignored. Aggregate statistics (means, SDs, null rates) are
  reproduced in `ecri/datagen.py` as facts about the data, not as a copy of it.

This repo asserts no rights over the ECRI data.
