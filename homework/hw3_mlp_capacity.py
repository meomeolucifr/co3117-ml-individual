"""HW3 Activity 5: controlled capacity experiment, two MLPs differing only in
hidden width, on the frozen HAR protocol (same split/seed as the rest of the semester).

Run: python homework/hw3_mlp_capacity.py
"""
import sys
import time
from pathlib import Path

from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_train  # noqa: E402

SEED = 2452879
WIDTHS = [16, 64]


def run():
    X, y, subjects = load_train()
    X_arr, y_arr = X.to_numpy(), y.to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    fit_idx, val_idx = next(splitter.split(X_arr, y_arr, groups=subjects))
    X_fit, y_fit = X_arr[fit_idx], y_arr[fit_idx]
    X_val, y_val = X_arr[val_idx], y_arr[val_idx]
    assert set(subjects.iloc[fit_idx]) & set(subjects.iloc[val_idx]) == set()

    for width in WIDTHS:
        clf = MLPClassifier(
            hidden_layer_sizes=(width,),
            random_state=SEED,
            max_iter=300,
        )
        start = time.perf_counter()
        clf.fit(X_fit, y_fit)
        elapsed = time.perf_counter() - start

        train_score = f1_score(y_fit, clf.predict(X_fit), average="macro")
        val_score = f1_score(y_val, clf.predict(X_val), average="macro")

        print(f"\n=== hidden_layer_sizes=({width},) ===")
        print(f"training Macro-F1: {train_score:.4f}")
        print(f"validation Macro-F1: {val_score:.4f}")
        print(f"runtime: {elapsed:.2f}s")
        print(f"n_iter_: {clf.n_iter_}, converged: {clf.n_iter_ < clf.max_iter}")


if __name__ == "__main__":
    run()
