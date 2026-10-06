"""HW2 Activity 5: decision-tree audit on the frozen HAR protocol.

Baseline tree (unrestricted depth) vs. a depth-limited tree, on the same
subject-aware split used throughout the semester (protocol.yaml, seed 2452879).

Run: python homework/hw2_tree_audit.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_train  # noqa: E402

SEED = 2452879
SMALLER_MAX_DEPTH = 5


def report(name, clf, X_fit, y_fit, X_val, y_val, feature_names):
    train_score = f1_score(y_fit, clf.predict(X_fit), average="macro")
    val_score = f1_score(y_val, clf.predict(X_val), average="macro")
    n_leaves = clf.get_n_leaves()
    depth = clf.get_depth()
    top3_idx = np.argsort(clf.feature_importances_)[::-1][:3]
    top3 = [(feature_names[i], round(clf.feature_importances_[i], 4)) for i in top3_idx]

    print(f"\n=== {name} ===")
    print(f"max_depth setting: {clf.max_depth}, actual depth reached: {depth}")
    print(f"number of leaves: {n_leaves}")
    print(f"training Macro-F1: {train_score:.4f}")
    print(f"validation Macro-F1: {val_score:.4f}")
    print(f"top 3 features by importance: {top3}")
    return train_score, val_score, n_leaves, depth


def run():
    X, y, subjects = load_train()
    # HAR's own feature names contain duplicates (a known quirk of the raw dataset);
    # fit on the underlying array and keep names separately for reporting.
    feature_names = list(X.columns)
    X_arr, y_arr = X.to_numpy(), y.to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    fit_idx, val_idx = next(splitter.split(X_arr, y_arr, groups=subjects))
    X_fit, y_fit = X_arr[fit_idx], y_arr[fit_idx]
    X_val, y_val = X_arr[val_idx], y_arr[val_idx]
    assert set(subjects.iloc[fit_idx]) & set(subjects.iloc[val_idx]) == set()

    baseline = DecisionTreeClassifier(max_depth=None, random_state=SEED)
    baseline.fit(X_fit, y_fit)
    report("Baseline (max_depth=None, unrestricted)", baseline, X_fit, y_fit, X_val, y_val, feature_names)

    smaller = DecisionTreeClassifier(max_depth=SMALLER_MAX_DEPTH, random_state=SEED)
    smaller.fit(X_fit, y_fit)
    report(f"Smaller tree (max_depth={SMALLER_MAX_DEPTH})", smaller, X_fit, y_fit, X_val, y_val, feature_names)


if __name__ == "__main__":
    run()
