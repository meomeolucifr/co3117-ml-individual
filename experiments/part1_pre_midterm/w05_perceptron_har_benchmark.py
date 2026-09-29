"""W05 benchmark: own OneVsRestPerceptron vs sklearn.linear_model.Perceptron on the
frozen HAR use case.

Uses the same subject-aware split as foundations_learning_curve.py (GroupShuffleSplit,
seed 2452879) on the training population only - the sealed test set stays untouched
per the frozen protocol (data/README.md).

Run: python experiments/part1_pre_midterm/w05_perceptron_har_benchmark.py
Outputs:
  - results/w05_perceptron_har_benchmark.csv
  - results/figures/w05_perceptron_har_confusion.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron as SkPerceptron
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import load_train  # noqa: E402
from src.from_scratch.perceptron import OneVsRestPerceptron  # noqa: E402

SEED = 2452879


def run():
    X, y, subjects = load_train()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    fit_idx, val_idx = next(splitter.split(X, y, groups=subjects))
    X_fit, y_fit = X.iloc[fit_idx].to_numpy(), y.iloc[fit_idx].to_numpy()
    X_val, y_val = X.iloc[val_idx].to_numpy(), y.iloc[val_idx].to_numpy()
    assert set(subjects.iloc[fit_idx]) & set(subjects.iloc[val_idx]) == set()

    results = []

    own = OneVsRestPerceptron(learning_rate=0.1, n_epochs=200)
    own.fit(X_fit, y_fit)
    own_pred = own.predict(X_val)
    results.append({
        "model": "own OneVsRestPerceptron",
        "macro_f1": f1_score(y_val, own_pred, average="macro"),
        "accuracy": accuracy_score(y_val, own_pred),
    })

    sk = SkPerceptron(max_iter=200, random_state=SEED)
    sk.fit(X_fit, y_fit)
    sk_pred = sk.predict(X_val)
    results.append({
        "model": "sklearn Perceptron",
        "macro_f1": f1_score(y_val, sk_pred, average="macro"),
        "accuracy": accuracy_score(y_val, sk_pred),
    })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    results_dir = ROOT / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "w05_perceptron_har_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # Failure analysis: own model's confusion matrix + 5 example misclassifications
    labels = sorted(set(y_val))
    cm = confusion_matrix(y_val, own_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Own OneVsRestPerceptron - confusion matrix (validation)")
    fig.tight_layout()
    fig_path = figures_dir / "w05_perceptron_har_confusion.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")

    wrong_idx = np.where(own_pred != y_val)[0]
    print(f"\n{len(wrong_idx)} / {len(y_val)} misclassified by own model on validation")
    print("5 example errors (true -> predicted):")
    rng = np.random.default_rng(SEED)
    sample = rng.choice(wrong_idx, size=min(5, len(wrong_idx)), replace=False)
    for i in sample:
        print(f"  true={y_val[i]:<20} predicted={own_pred[i]}")

    # Most common confusion pair
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    i, j = np.unravel_index(off_diag.argmax(), off_diag.shape)
    print(f"\nMost common confusion: true={labels[i]} predicted as {labels[j]} "
          f"({off_diag[i, j]} times)")


if __name__ == "__main__":
    run()
