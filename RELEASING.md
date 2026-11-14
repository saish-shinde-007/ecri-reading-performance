# Releasing

A release tag certifies that a fresh clone reproduces every number the
project reports. The check is one command, so cutting a release is mostly
deciding that it is time.

## Versions

| Tag | Meaning |
|---|---|
| `v0.9.0` | Our tickets complete (T01–T28, T39): the pipeline, the evaluation, the model and its ranking contract, and this reproducibility gate. 70% of the plan. |
| `v1.0.0` | All 39 tickets complete, including Sarvesh's T29–T38 (issues #2–#11). |

`v1.0.0` waits for the final report (T38), because the release is meant to
certify the numbers that report quotes.

## Cutting a release

1. **Everything for the release is merged to `main`**, and CI on `main` is
   green, including the `reproduce` job.
2. **A fresh environment:**
   ```bash
   git clone https://github.com/saish-shinde-007/ecri-reading-performance.git
   cd ecri-reading-performance
   make install    # exactly uv.lock
   make data       # pinned commit, sha256-verified
   ```
3. **Run the gate. It must pass end to end:**
   ```bash
   make release-check
   ```
   Lint, the full test suite, the data checksums, the label rule, the README
   check, and `ecri reproduce` against `results/expected.json`.
4. **If a ticket changed a published number on purpose** — a new model in
   the comparison, a fixed bug — re-record it deliberately and say why:
   ```bash
   .venv/bin/python -m ecri.cli reproduce --write   # prints which numbers moved
   ```
   Then explain every moved number in `docs/findings.md`, commit the record,
   the README and the finding together, and run `make release-check` again.
   A number that moved without an explanation blocks the release.
5. **Bump the version** in `pyproject.toml` and `src/ecri/__init__.py`, then
   `uv lock` so the lockfile agrees, and commit.
6. **Tag the merge commit on `main` and push the tag:**
   ```bash
   git tag -a v1.0.0 -m "v1.0.0: all 39 tickets; make release-check passes"
   git push origin v1.0.0
   ```

## Adding a model to the gate

When a new model should appear in the README table (T29's boosting models,
T32's MaskMLP), add its registry name to `CANONICAL_MODELS` in
`src/ecri/reproduce.py` and its display name to `DISPLAY`. Then run
`reproduce --write`, which records it and regenerates the README table. Until
then, `ecri reproduce` reports the new model's numbers as "not in the record",
and `test_the_record_covers_every_canonical_model` fails, so a model can't be
added to the gate and left out of the record.
