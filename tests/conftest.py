import pytest

from ecri import clean, datagen, split

# Small enough to keep the suite fast, large enough to keep 44 schools and the
# label subsample meaningful.
ROWS = 2500
SEED = 7


@pytest.fixture(scope="session")
def raw_frame():
    return datagen.generate_frame(n_rows=ROWS, seed=SEED)


@pytest.fixture(scope="session")
def prepared(raw_frame):
    frame, _ = clean.prepare(raw_frame, "RMwidRS_target")
    return frame


@pytest.fixture(scope="session")
def split_data(prepared):
    return split.make_split(prepared, "RMwidRS_target", strategy="school", seed=SEED)
