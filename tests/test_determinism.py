"""The seed audit (T39), as tests rather than a one-off review.

Every result in this repo must be a function of the data, the lockfile and
config.seed — nothing else. These tests fail the build when that stops being
true, instead of waiting for a number to quietly change.
"""

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ecri import evaluate

SRC = Path(__file__).resolve().parents[1] / "src" / "ecri"

# Constructors that draw from unseeded or global random state.
UNSEEDED = [
    (r"default_rng\(\s*\)", "default_rng() with no seed"),
    (r"random_state\s*=\s*None", "random_state=None"),
    (
        r"np\.random\.(seed|rand|randn|randint|random|choice|shuffle|permutation)\(",
        "numpy's global random state",
    ),
    (r"^\s*import random\b|^\s*from random import", "the stdlib random module"),
]


def test_no_unseeded_randomness_in_the_package():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for pattern, why in UNSEEDED:
                if re.search(pattern, line):
                    offenders.append(f"{path.relative_to(SRC)}:{lineno}: {why}")
    assert not offenders, "unseeded randomness:\n" + "\n".join(offenders)


# Covers every estimator family, including the multithreaded ones (random
# forest uses 2 jobs; gradient boosting and the MLP use threaded maths).
FAMILIES = ["logistic", "random_forest", "hist_gradient_boosting", "mlp"]


def test_same_seed_gives_identical_out_of_fold_scores(prepared):
    first, _ = evaluate.run_models(
        prepared, "RMwidRS_target", FAMILIES, repeats=2, n_folds=4, seed=11
    )
    second, _ = evaluate.run_models(
        prepared, "RMwidRS_target", FAMILIES, repeats=2, n_folds=4, seed=11
    )
    for name in FAMILIES:
        assert np.array_equal(first[name].oof_scores, second[name].oof_scores), name


def test_a_different_seed_changes_the_folds(prepared):
    a, _ = evaluate.run_models(
        prepared, "RMwidRS_target", ["logistic"], repeats=1, n_folds=4, seed=11
    )
    b, _ = evaluate.run_models(
        prepared, "RMwidRS_target", ["logistic"], repeats=1, n_folds=4, seed=12
    )
    assert not np.array_equal(a["logistic"].oof_scores, b["logistic"].oof_scores)


SCRIPT = """
import hashlib, warnings
warnings.simplefilter("ignore")
from ecri import clean, datagen, evaluate
frame, _ = clean.prepare(datagen.generate_frame(n_rows=1500, seed=3), "RMwidRS_target")
results, _ = evaluate.run_models(frame, "RMwidRS_target", ["logistic", "random_forest"],
                                 repeats=1, n_folds=4, seed=5)
digest = hashlib.sha256()
for name in sorted(results):
    digest.update(results[name].oof_scores.round(12).tobytes())
print(digest.hexdigest())
"""


@pytest.mark.parametrize("hashseed", ["0", "4242"])
def test_results_do_not_depend_on_python_hash_order(hashseed):
    """Set and dict iteration order varies with PYTHONHASHSEED; results must not."""

    def run(seed: str) -> str:
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        return out.stdout.strip()

    assert run(hashseed) == run("12345")


def test_digest_helper_is_stable():
    # Guards the test above against a vacuous pass: the digest must depend on data.
    a = hashlib.sha256(np.array([0.1, 0.2]).tobytes()).hexdigest()
    b = hashlib.sha256(np.array([0.1, 0.3]).tobytes()).hexdigest()
    assert a != b
