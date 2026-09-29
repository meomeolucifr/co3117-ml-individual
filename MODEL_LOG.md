# MODEL_LOG.md - per-model record

One section per model family. Fill in as each week's work is committed.

## Template

```
## <Model name> (Syllabus Ch. <n>, Depth <A/B/C>)

- **Objective / factorization / decision rule:**
- **Assumptions, inductive bias, expected failure modes:**
- **Data representation & preprocessing:**
- **Own pre-reference commit:** <hash>
- **Reference code/notebook consulted:** <exact file/URL>
- **Hyperparameters & selection procedure:**
- **Metrics:** Macro-F1 = , Accuracy = , runtime =
- **Diagnostic plot/table:** results/figures/<file>
- **Focused experiment:**
- **Error/limitation analysis & use-case fit:**
- **Exam-ready paragraph (no code):**
```

---

(Entries below will be appended weekly, oldest first.)

## Perceptron (Syllabus Ch. 3, Depth A)

- **Objective / decision rule:** Binary linear classifier. $z = \mathbf{w}\cdot\mathbf{x}+b$,
  $\hat y=\mathrm{sign}(z)$. Error-driven update on misclassification only:
  $\mathbf{w}\leftarrow\mathbf{w}+\eta y_i\mathbf{x}_i$, $b\leftarrow b+\eta y_i$.
  Multi-class (6 HAR activities) handled via one-vs-rest: one binary Perceptron per
  class, prediction = `argmax` over the classifiers' continuous net-input scores (not
  their signs), to resolve ties and all-negative cases.
- **Assumptions, inductive bias, expected failure modes:** Assumes the two classes are
  linearly separable; convergence (Perceptron Convergence Theorem) is only guaranteed in
  that case. On non-separable data it oscillates indefinitely instead of converging.
- **Data representation & preprocessing:** Frozen HAR 561-feature vectors (already
  normalized per UCI's own preprocessing), subject-aware fit/validation split
  (`GroupShuffleSplit`, 75/25, seed 2452879) carved out of the training population only
  - the sealed test set was not touched for this benchmark.
- **Own pre-reference commit:** `4eaf678`
- **Reference code/notebook consulted:** ML-From-Scratch,
  `mlfromscratch/deep_learning/perceptron.py` (pasted into this session, not fetched
  from the repo directly) - discovered it implements full-batch gradient descent with a
  configurable Sigmoid activation + SquareLoss by default, i.e. a Delta-rule learner,
  not the discrete Perceptron rule despite the class name.
- **Hyperparameters & selection procedure:** `learning_rate=0.1`, `n_epochs` capped at
  200 (own model); `sklearn.linear_model.Perceptron(max_iter=200)` with matching seed.
  No hyperparameter search yet - default/matched settings only.
- **Metrics (validation split, n=2060):**

  | Model | Macro-F1 | Accuracy |
  |---|---|---|
  | Own `OneVsRestPerceptron` | 0.9633 | 0.9636 |
  | `sklearn.linear_model.Perceptron` | 0.9734 | 0.9738 |

  Script: `experiments/part1_pre_midterm/w05_perceptron_har_benchmark.py`; data:
  `results/w05_perceptron_har_benchmark.csv`.
- **Diagnostic plot/table:** `results/figures/w05_perceptron_har_confusion.png`
  (own model's confusion matrix); `results/figures/w05_perceptron_convergence.png`,
  `results/w05_perceptron_convergence.csv` (convergence experiment, toy data - see below).
- **Focused experiment:** Number of flipped (noisy) labels vs. convergence, on a
  synthetic linearly-separable 2-blob set (120 samples, seed 2452879). Clean set (0
  flips): converged in 2 epochs, 100% train accuracy. Any noise (1+ flips): never
  converged within the 200-epoch cap, train accuracy stayed high (83-99%) but
  non-monotonic in the noise level, matching the "oscillates forever on non-separable
  data" prediction. Script: `experiments/part1_pre_midterm/w05_perceptron_convergence.py`.
- **Error/limitation analysis & use-case fit:** 75/2060 validation samples misclassified
  (3.6% error) by the own model. The single largest confusion pair is
  **SITTING predicted as STANDING** (34 of 75 errors, ~45%) - confirms the expected
  limitation: both are static postures with overlapping accelerometer/gyroscope
  signatures, so a single linear hyperplane per one-vs-rest class struggles to separate
  them, while the more dynamic activities (WALKING variants) are separated cleanly.
  Fit for use case: strong as a linear baseline (>96% Macro-F1) but not expected to be
  the final model precisely because of this static-posture ambiguity - later chapters
  (SVM with kernels, MLP) should be checked for whether nonlinearity closes this gap.
- **Exam-ready paragraph (no code):** The Perceptron learns a linear decision boundary
  by nudging its weights only when it misclassifies a training example, moving the
  boundary toward that example's correct side. It provably converges in finite steps if
  and only if the two classes are linearly separable; on non-separable data it
  oscillates forever instead of settling. Extending it to multiple classes requires an
  explicit strategy such as one-vs-rest, since the base algorithm is inherently binary.
  On the HAR use case this linear model already reaches over 96% Macro-F1, but its
  errors concentrate on distinguishing SITTING from STANDING - two postures a straight
  decision boundary per class struggles to tell apart.

W05 complete: own implementation, dissection, convergence experiment, and HAR benchmark
all done. Tag `w05` at commit time.
