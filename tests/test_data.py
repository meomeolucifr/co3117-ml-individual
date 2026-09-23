"""Sanity checks for the frozen data protocol (subject-disjoint split, no leakage)."""

from src.data import load_test, load_train


def test_no_subject_overlap():
    _, _, subj_train = load_train()
    _, _, subj_test = load_test()
    assert set(subj_train) & set(subj_test) == set()


def test_expected_classes():
    _, y_train, _ = load_train()
    expected = {"WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"}
    assert set(y_train.unique()) == expected


def test_no_missing_values():
    X_train, _, _ = load_train()
    assert not X_train.isnull().values.any()
