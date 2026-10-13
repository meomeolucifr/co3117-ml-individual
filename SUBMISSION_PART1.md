# SUBMISSION PART 1

- Tag: part1-final
- Commit (full hash): TODO - fill in with the exact commit this tag points to, right before tagging
- Report: report/part1_summary.pdf
- Two-A4 exam sheet: exam/a4-notes-part1-midterm.pdf
- Reproduce: `python -m venv .venv && pip install -r requirements.txt`, then
  `python tools/check.py` and the individual experiment scripts under
  `experiments/part1_pre_midterm/` (each one is self-contained and reads the frozen
  split from `protocol.yaml` / `src/data.py`, seed 2452879 throughout).
- Release-day baseline and catch-up: exercises/release-baseline-w01-w02.pdf,
  docs/pre-release/PRE_RELEASE_CATCHUP.md
- Frozen protocol: protocol.yaml; weekly posts W05-W07 and drills linked from PROGRESS.md
- Known limitations and errata (corrections found after 14 Oct go to Part II, never into
  Part I history):
  - GaussianNB's conditional-independence assumption is visibly violated on HAR's
    correlated 561-feature set (Macro-F1 0.768 vs. the Perceptron baseline's 0.963);
    kept as the required Depth-A generative baseline, not as a competitive final model.
  - The GA feature-selection search (W07) uses `sklearn.naive_bayes.GaussianNB` as the
    fitness-evaluation model for speed; the from-scratch `GaussianNB` was verified to
    match it exactly (100% prediction agreement) in the W06 benchmark, so this does not
    change what the search is actually optimizing against.
