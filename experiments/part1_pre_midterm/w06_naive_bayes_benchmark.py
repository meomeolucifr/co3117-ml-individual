"""W06 benchmark: own GaussianNB vs sklearn.naive_bayes.GaussianNB on the frozen
HAR protocol. Same subject-aware split as previous weeks (seed 2452879).

Run: python experiments/part1_pre_midterm/w06_naive_bayes_benchmark.py
"""
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB as SkGaussianNB

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import load_train  # noqa: E402
from src.from_scratch.naive_bayes import GaussianNB  # noqa: E402

SEED = 2452879


def run():
    X, y, subjects = load_train()
    X_arr, y_arr = X.to_numpy(), y.to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    fit_idx, val_idx = next(splitter.split(X_arr, y_arr, groups=subjects))
    X_fit, y_fit = X_arr[fit_idx], y_arr[fit_idx]
    X_val, y_val = X_arr[val_idx], y_arr[val_idx]
    assert set(subjects.iloc[fit_idx]) & set(subjects.iloc[val_idx]) == set()

    results = []

    t0 = time.perf_counter()
    own = GaussianNB()
    own.fit(X_fit, y_fit)
    own_pred = own.predict(X_val)
    own_time = time.perf_counter() - t0
    results.append({
        "model": "own GaussianNB",
        "macro_f1": f1_score(y_val, own_pred, average="macro"),
        "accuracy": accuracy_score(y_val, own_pred),
        "runtime_s": own_time,
    })

    t0 = time.perf_counter()
    sk = SkGaussianNB()
    sk.fit(X_fit, y_fit)
    sk_pred = sk.predict(X_val)
    sk_time = time.perf_counter() - t0
    results.append({
        "model": "sklearn GaussianNB",
        "macro_f1": f1_score(y_val, sk_pred, average="macro"),
        "accuracy": accuracy_score(y_val, sk_pred),
        "runtime_s": sk_time,
    })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    results_dir = ROOT / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "w06_naive_bayes_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # Agreement between own and sklearn predictions (should be near-identical since
    # predict_log_proba matched to atol=1e-4 in the contract test)
    agreement = np.mean(own_pred == sk_pred)
    print(f"\nOwn vs sklearn prediction agreement: {agreement:.4f}")

    # Failure analysis: own model's confusion matrix
    labels = sorted(set(y_val))
    cm = confusion_matrix(y_val, own_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Own GaussianNB - confusion matrix (validation)")
    fig.tight_layout()
    fig_path = figures_dir / "w06_naive_bayes_confusion.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")

    wrong_idx = np.where(own_pred != y_val)[0]
    print(f"\n{len(wrong_idx)} / {len(y_val)} misclassified by own model on validation")
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    i, j = np.unravel_index(off_diag.argmax(), off_diag.shape)
    print(f"Most common confusion: true={labels[i]} predicted as {labels[j]} "
          f"({off_diag[i, j]} times)")


if __name__ == "__main__":
    run()
