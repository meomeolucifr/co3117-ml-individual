"""Foundations (Ch.1) diagnostic: train vs validation Macro-F1 as model capacity
(Decision Tree max_depth) increases, on the frozen HAR training population.

Uses a subject-aware split (GroupKFold-style single split on subject_train.txt) so
no subject appears in both the fit and validation portions, per data/README.md.

Run: python experiments/part1_pre_midterm/foundations_learning_curve.py
Outputs:
  - results/foundations_learning_curve.csv
  - results/figures/foundations_learning_curve.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import load_train  # noqa: E402

SEED = 2452879
DEPTHS = list(range(1, 21))


def run():
    X, y, subjects = load_train()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    fit_idx, val_idx = next(splitter.split(X, y, groups=subjects))
    X_fit, y_fit = X.iloc[fit_idx], y.iloc[fit_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    assert set(subjects.iloc[fit_idx]) & set(subjects.iloc[val_idx]) == set()

    rows = []
    for depth in DEPTHS:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
        clf.fit(X_fit, y_fit)
        train_f1 = f1_score(y_fit, clf.predict(X_fit), average="macro")
        val_f1 = f1_score(y_val, clf.predict(X_val), average="macro")
        rows.append({"max_depth": depth, "train_macro_f1": train_f1, "val_macro_f1": val_f1})
        print(f"depth={depth:>2} | train Macro-F1={train_f1:.4f} | val Macro-F1={val_f1:.4f}")

    df = pd.DataFrame(rows)
    results_dir = ROOT / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "foundations_learning_curve.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    gap = df["train_macro_f1"] - df["val_macro_f1"]
    best_depth = df.loc[df["val_macro_f1"].idxmax(), "max_depth"]
    print(f"Best val Macro-F1 at max_depth={best_depth}")
    print(f"Train-val gap at max depth tested: {gap.iloc[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["max_depth"], df["train_macro_f1"], marker="o", label="Train Macro-F1")
    ax.plot(df["max_depth"], df["val_macro_f1"], marker="o", label="Validation Macro-F1")
    ax.axvline(best_depth, color="gray", linestyle="--", alpha=0.6, label=f"Best val depth={best_depth}")
    ax.set_xlabel("Decision Tree max_depth (model capacity)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Foundations diagnosis: capacity vs train/validation Macro-F1 (HAR)")
    ax.legend()
    ax.set_xticks(DEPTHS)

    fig_path = figures_dir / "foundations_learning_curve.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    run()
