"""W07 / HW5 Activity 5: bounded GA search for feature selection on the frozen HAR
protocol. Runs 3 independent seeds, same subject-aware fit/validation split for all
(seed 2452879 for the split itself, per protocol.yaml).

Run: python experiments/part1_pre_midterm/w07_ga_feature_selection.py
"""
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import load_train  # noqa: E402
from src.from_scratch.genetic_algorithm import GeneticFeatureSelection  # noqa: E402

SPLIT_SEED = 2452879
GA_SEEDS = [2452879, 1, 2]

RESULTS_DIR = ROOT / "results"
CSV_PATH = RESULTS_DIR / "w07_ga_feature_selection.csv"
HISTORY_PATH = RESULTS_DIR / "w07_ga_feature_selection_history.json"


def _load_progress():
    """Resume support: a previous run may have crashed partway (e.g. low system
    memory) after completing some seeds. Reuse those instead of re-running them."""
    rows, histories = [], {}
    if CSV_PATH.exists() and HISTORY_PATH.exists():
        rows = pd.read_csv(CSV_PATH).to_dict("records")
        histories = {int(k): v for k, v in json.loads(HISTORY_PATH.read_text()).items()}
    return rows, histories


def run():
    X, y, subjects = load_train()
    X_arr, y_arr = X.to_numpy(), y.to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SPLIT_SEED)
    fit_idx, val_idx = next(splitter.split(X_arr, y_arr, groups=subjects))
    X_fit, y_fit = X_arr[fit_idx], y_arr[fit_idx]
    X_val, y_val = X_arr[val_idx], y_arr[val_idx]
    assert set(subjects.iloc[fit_idx]) & set(subjects.iloc[val_idx]) == set()

    RESULTS_DIR.mkdir(exist_ok=True)
    rows, histories = _load_progress()
    done_seeds = {r["seed"] for r in rows}

    for seed in GA_SEEDS:
        if seed in done_seeds:
            print(f"seed={seed}: already done (resumed from disk), skipping")
            continue

        t0 = time.perf_counter()
        ga = GeneticFeatureSelection(
            population_size=50, n_generations=100, patience=20, seed=seed,
        )
        ga.fit(X_fit, y_fit, X_val, y_val)
        elapsed = time.perf_counter() - t0

        rows.append({
            "seed": seed,
            "best_fitness": ga.best_fitness_,
            "n_features_selected": ga.n_features_selected_,
            "best_generation": ga.best_generation_,
            "n_generations_run": ga.n_generations_run_,
            "stopping_reason": ga.stopping_reason_,
            "runtime_s": round(elapsed, 1),
        })
        histories[seed] = ga.history_
        print(f"seed={seed}: best_fitness={ga.best_fitness_:.4f}, "
              f"n_features={ga.n_features_selected_}, "
              f"gens={ga.n_generations_run_} ({ga.stopping_reason_}), "
              f"runtime={elapsed:.1f}s", flush=True)

        # Persist after every seed so a later crash (e.g. low system memory) loses
        # at most one seed's work, and free the fitted estimator before the next seed.
        pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
        HISTORY_PATH.write_text(json.dumps(histories))
        del ga
        gc.collect()

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    print(f"\nSaved {CSV_PATH}")

    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Convergence plot: best-so-far fitness per generation, one line per seed
    fig, ax = plt.subplots(figsize=(7, 5))
    for seed, hist in histories.items():
        gens = [h["generation"] for h in hist]
        best = [h["best_fitness"] for h in hist]
        ax.plot(gens, best, marker=".", label=f"seed={seed}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best-so-far fitness (Macro-F1 - penalty)")
    ax.set_title("GA feature-selection convergence across 3 seeds")
    ax.legend()
    fig.tight_layout()
    fig_path = figures_dir / "w07_ga_convergence.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")

    spread = df["best_fitness"].max() - df["best_fitness"].min()
    print(f"\nBest-fitness spread across seeds: {spread:.4f} "
          f"(min={df['best_fitness'].min():.4f}, max={df['best_fitness'].max():.4f})")


if __name__ == "__main__":
    run()
