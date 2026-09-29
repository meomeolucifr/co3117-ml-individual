import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.from_scratch.perceptron import Perceptron  # noqa: E402

SEED = 2452879
N_EPOCHS_CAP = 200
LEARNING_RATE = 0.1
NOISE_LEVELS = [0, 1, 2, 3, 5, 8, 12, 20]


def make_separable_dataset(n_per_class: int = 60, seed: int = SEED):
    """Two well-separated 2D Gaussian blobs, labels in {-1, +1}."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(loc=(3, 3), scale=0.8, size=(n_per_class, 2))
    neg = rng.normal(loc=(-3, -3), scale=0.8, size=(n_per_class, 2))
    X = np.vstack([pos, neg])
    y = np.array([1] * n_per_class + [-1] * n_per_class)
    return X, y


def flip_labels(y: np.ndarray, n_flip: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed + n_flip)  # distinct but reproducible per level
    y_noisy = y.copy()
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    y_noisy[flip_idx] *= -1
    return y_noisy


def run():
    X, y_clean = make_separable_dataset()
    rows = []

    for n_flip in NOISE_LEVELS:
        y_noisy = flip_labels(y_clean, n_flip)
        model = Perceptron(learning_rate=LEARNING_RATE, n_epochs=N_EPOCHS_CAP)
        model.fit(X, y_noisy)
        train_acc = np.mean(model.predict(X) == y_noisy)
        rows.append({
            "n_flipped_labels": n_flip,
            "converged": model.converged,
            "n_epochs_run": model.n_epochs_run,
            "train_accuracy": round(float(train_acc), 4),
        })
        print(f"flipped={n_flip:>3} | converged={model.converged!s:5} | "
              f"epochs={model.n_epochs_run:>3} | train_acc={train_acc:.3f}")

    results_dir = ROOT / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "w05_perceptron_convergence.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {csv_path}")

    fig, ax1 = plt.subplots(figsize=(6, 4))
    flips = [r["n_flipped_labels"] for r in rows]
    epochs = [r["n_epochs_run"] for r in rows]
    converged = [r["converged"] for r in rows]

    colors = ["tab:green" if c else "tab:red" for c in converged]
    ax1.bar(flips, epochs, color=colors, width=0.8)
    ax1.set_xlabel("Number of flipped labels (out of 120 samples)")
    ax1.set_ylabel(f"Epochs run (cap = {N_EPOCHS_CAP})")
    ax1.set_title("Perceptron convergence vs. label noise (green=converged, red=hit cap)")
    ax1.set_xticks(flips)

    fig_path = figures_dir / "w05_perceptron_convergence.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    run()
