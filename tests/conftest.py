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


# ── T27/T28 fixtures. Reusable by app tests (T37): a trained, saved bundle and
# students it never saw. Synthetic throughout, so they run without real data.


@pytest.fixture(scope="session")
def bundle(prepared):
    from ecri import persist

    return persist.train_bundle(
        prepared,
        "RMwidRS_target",
        "logistic",
        repeats=1,
        n_folds=4,
        seed=SEED,
        data_info={"source": "synthetic"},
    )


@pytest.fixture(scope="session")
def bundle_dir(bundle, tmp_path_factory):
    from ecri import persist

    return persist.save_bundle(bundle, tmp_path_factory.mktemp("bundle") / "logistic")


@pytest.fixture()
def unseen_records():
    """Students from a different generator seed, outcomes removed."""
    import pandas as pd

    from ecri import schema

    frame, _ = clean.prepare(
        datagen.generate_frame(n_rows=1500, seed=SEED + 1000), "RMwidRS_target"
    )
    X, _, _ = split.features_and_target(frame, "RMwidRS_target")
    return pd.concat([frame[[schema.STUDENT_ID]], X], axis=1).reset_index(drop=True)
