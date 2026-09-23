"""Loader for the UCI HAR Dataset, frozen at R0 (see data/README.md)."""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "UCI HAR Dataset"
SEED = 2452879


def _load_split(split: str):
    split_dir = RAW_DIR / split
    X = pd.read_csv(split_dir / f"X_{split}.txt", sep=r"\s+", header=None)
    y = pd.read_csv(split_dir / f"y_{split}.txt", sep=r"\s+", header=None).iloc[:, 0]
    subjects = pd.read_csv(split_dir / f"subject_{split}.txt", sep=r"\s+", header=None).iloc[:, 0]

    feature_names = pd.read_csv(RAW_DIR / "features.txt", sep=r"\s+", header=None)[1]
    X.columns = feature_names

    activity_labels = pd.read_csv(RAW_DIR / "activity_labels.txt", sep=r"\s+", header=None, index_col=0)[1]
    y_named = y.map(activity_labels)

    return X, y_named, subjects


def load_train():
    """Returns (X_train, y_train, subject_train). Subject-disjoint from test."""
    return _load_split("train")


def load_test():
    """Returns (X_test, y_test, subject_test). Sealed until final comparison."""
    return _load_split("test")


if __name__ == "__main__":
    X_train, y_train, subj_train = load_train()
    X_test, y_test, subj_test = load_test()
    print("train:", X_train.shape, "subjects:", subj_train.nunique())
    print("test:", X_test.shape, "subjects:", subj_test.nunique())
    print("overlap subjects train/test:", set(subj_train) & set(subj_test))
    print("classes:", sorted(y_train.unique()))
