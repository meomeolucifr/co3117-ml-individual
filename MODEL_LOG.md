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

## Decision Tree split routine (Syllabus Ch. 2, release catch-up, Depth B)

- **Objective / decision rule:** One impurity/split routine, not a full tree: `entropy(y)`
  (Shannon entropy in bits), `gini(y)`, `information_gain(y, mask)` (entropy of `y` minus
  the sample-weighted entropy of the two `mask` partitions), and `best_threshold(x, y)`
  (best `x <= t` split on one continuous feature by information gain, `t` a midpoint
  between consecutive distinct values).
- **Assumptions, inductive bias, expected failure modes:** Greedy, single-feature
  threshold search; `information_gain` must skip an empty partition's entropy term
  (undefined otherwise) rather than crash.
- **Data representation & preprocessing:** Works on raw label arrays / one continuous
  feature column; no HAR-specific preprocessing needed for the routine itself.
- **Own pre-reference commit:** written and verified against `tests/contract/test_tree_split.py`
  before any reference code was consulted.
- **Hyperparameters & selection procedure:** n/a (deterministic routine, no hyperparameters).
- **Metrics:** Contract tests pass (`entropy`/`gini` on known distributions, perfect vs.
  useless split `information_gain`, `best_threshold` matched against brute force within
  1e-9).
- **Error/limitation analysis & use-case fit:** Direct building block for the Decision
  Tree catch-up work (release-baseline); same entropy/IG numbers already verified by
  hand in `docs/pre-release/PRE_RELEASE_CATCHUP.md`'s Play Tennis worked example.
- **Exam-ready paragraph (no code):** Entropy and Gini both measure node impurity;
  Information Gain is the parent's impurity minus the sample-weighted impurity of the
  children after a split, and greedy tree induction picks the split that maximizes it.
  For continuous attributes, only midpoints between consecutive distinct sorted values
  need to be checked, since the information gain can only change at those points.

## MLP (Syllabus Ch. 3, Depth A)

- **Objective / decision rule:** `L`-layer feedforward network, `Z_l = A_{l-1}W_l + b_l`,
  hidden activation **tanh**, output layer softmax, loss = mean cross-entropy.
  Backpropagation via the standard softmax+cross-entropy simplification
  $\partial L/\partial Z_L = (P-Y)/n$, then layer-by-layer chain rule with the tanh
  derivative $1-A_l^2$.
- **Assumptions, inductive bias, expected failure modes:** Differentiable activation
  required for exact gradients; dead/zero-gradient regions still possible far from the
  origin (vanishing gradient for saturated tanh units).
- **Data representation & preprocessing:** Not yet run on HAR; validated so far only on
  the contract tests' synthetic data (numerical gradient check).
- **Own pre-reference commit:** written and gradient-checked against
  `tests/contract/test_mlp.py` before any reference code was consulted.
- **Hyperparameters & selection procedure:** Weight init N(0, 0.1^2), bias init 0
  (seeded via `np.random.default_rng`); hidden sizes not yet chosen for HAR.
- **Metrics:** Contract test `test_backward_matches_numerical_gradient` passes with
  relative error < 1e-5 for both a 1-hidden-layer and a 2-hidden-layer architecture.
- **Error/limitation analysis & use-case fit:** First implementation used ReLU with
  He-scaled init ($\sqrt{2/n_{in}}$); this **failed** the 2-hidden-layer gradient check
  (relative error 0.19 on `b2`). Root cause: zero-initialized biases + ReLU meant that
  when an entire sample's hidden-1 output was exactly zero (all 6 neurons dead for that
  sample), layer 2's pre-activation for that sample landed exactly on the ReLU kink
  (`Z2 = 0@W2 + b2 = 0`) for every output neuron, so numerical and analytic gradients
  disagreed right at the non-differentiable point. Switching the hidden activation to
  tanh (smooth everywhere, no kink) fixed it without touching the chain-rule logic -
  recorded here as the week's failure/misconception since it's a general ReLU +
  zero-bias-init gradient-check trap, not specific to this network.
- **Exam-ready paragraph (no code):** An MLP is a stack of linear layers separated by a
  nonlinearity, ending in softmax for multi-class output. Backpropagation computes the
  output-layer error signal as `P - Y` (a clean consequence of pairing softmax with
  cross-entropy), then propagates it backward through each layer via the chain rule,
  multiplying by the local activation's derivative at each hidden layer. ReLU's
  non-differentiable kink at exactly zero can make numerical gradient checks unreliable
  in degenerate cases (e.g. an entire layer dead for one sample under zero-bias init);
  a smooth activation like tanh avoids that failure mode entirely.

W05 complete: Perceptron, tree-split routine, and MLP all implemented, dissected, and
passing their contract tests; HAR benchmark done for the Perceptron.
