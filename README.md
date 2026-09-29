# CO3117 - Individual Longitudinal Assignment (Two-Part Version)

**Student:** Nguyễn Thiện Nhân - 2452879
**Course:** CO3117 Machine Learning, HK261
**Design:** One dataset, one use case, many models - release-adapted for HK261 (release: Course Week 5, 23 Sep 2026).

## Use case

Predict a person's current physical activity from smartphone accelerometer/gyroscope
features (UCI Human Activity Recognition Using Smartphones). See [data/README.md](data/README.md)
for the frozen dataset version, split policy, metric, and seed.

## Repository map

| Path | Purpose |
|---|---|
| `PROGRESS.md` | Weekly instructor dashboard (one row per Course Week) |
| `MODEL_LOG.md` | Per-model record (objective, depth, hyperparameters, results) |
| `AI_USE.md` | Mandatory AI-assisted inquiry-based learning log |
| `REFERENCES.md` | Every source actually cited/used |
| `SUBMISSION_PART1.md` / `SUBMISSION_PART2.md` | Graded-submission checklists |
| `docs/pre-release/` | W01–W04 retrospective catch-up post |
| `docs/weekly/` | One post per Course Week (W05–W15) |
| `exercises/` | Scanned/PDF handwritten first-attempt drills + corrections |
| `exam/` | Two-A4 living exam sheet, midterm/final mock artifacts |
| `src/` | `data.py`, `metrics.py`, `from_scratch/`, `reference_adapters/` |
| `experiments/` | `part1_pre_midterm/`, `part2_post_midterm/` |
| `tests/` | Sanity/unit tests for from-scratch implementations |
| `results/` | `metrics.csv`, `figures/` |
| `report/` | `part1_summary.pdf`, `part2_final_report.pdf` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Then follow [data/README.md](data/README.md) to download and unzip the dataset into `data/raw/`.

## Git conventions

- Two-state commits from W03 onward: `first attempt` → `corrected/extended` after
  reference-code inspection / AI-assisted checking.
- Commit message pattern: `[W05][theory] ...`, `[W05][code] ...`, `[W05][review] ...`.
- Tags: `release-baseline` (R0), `w05`...`w15` (weekly, `w08-midterm` for midterm week),
  `part1-final` (14 Oct 2026), `part2-final` (final-exam minus 2 days).
- No back-dated commits or fabricated W01–W04 history - see [docs/pre-release/PRE_RELEASE_CATCHUP.md](docs/pre-release/PRE_RELEASE_CATCHUP.md).
