# W06 - Naive Bayes

> DRAFT - rewrite in your own words before this counts as submission evidence
> (Section 12.3). Grounded in real results from this week's work.

## A. Concept capsule

Naive Bayes is a generative classifier: it models the joint distribution
$P(X,Y) = P(Y)\prod_i P(X_i|Y)$ under the "naive" assumption that features are
conditionally independent given the class, then applies Bayes' rule to get
$P(Y|X) \propto P(Y)\prod_i P(X_i|Y)$. GaussianNB models each feature's class-conditional
distribution as $\mathcal{N}(\mu_{c,i}, \sigma^2_{c,i})$, estimated per class from the
training data (mean and variance), with `var_smoothing` adding a small constant to every
variance to avoid division by (near-)zero.

## B. One derivation or worked example

From the drill: $P(Y{=}1)=0.3$, $P(X{=}1|Y{=}1)=0.8$, $P(X{=}1|Y{=}0)=0.2$.
$P(X{=}1) = 0.8(0.3) + 0.2(0.7) = 0.38$.
$P(Y{=}1|X{=}1) = 0.8(0.3)/0.38 \approx 0.632$.

## C. Code-to-theory trace

- Per-class mean/variance ($\mu_{c,i}$, $\sigma^2_{c,i}$): `src/from_scratch/naive_bayes.py`,
  `GaussianNB.fit`, `self.theta_` / `self.var_`.
- Gaussian log-likelihood summed over features (naive independence assumption):
  `GaussianNB.predict_log_proba`, the `log_likelihood` computation.
- Numerically stable normalization (log-sum-exp) to turn joint log-likelihoods into a
  proper log-posterior: same method, the `max_log`/`log_norm` lines.

## D. One controlled experiment

Question: how sensitive is GaussianNB to `var_smoothing` on the HAR data? Swept
`var_smoothing` over 5 orders of magnitude on the frozen subject-aware split
(fit/val from `GroupShuffleSplit`, seed 2452879), everything else fixed.

| var_smoothing | Macro-F1 |
|---|---|
| 1e-9 (default) | 0.7679 |
| 1e-6 | 0.8100 |
| 1e-3 | **0.8411** |
| 1e-1 | 0.8172 |
| 1.0 | 0.8077 |

Result: far from negligible - the default `var_smoothing=1e-9` is actually the worst
setting tested. Performance improves sharply as smoothing increases, peaks around
`1e-3` (Macro-F1 0.841, +0.073 over default), then degrades again at `var_smoothing=1.0`
as the per-feature variances get swamped by the smoothing term and the model loses
discriminative power between classes. This is a bias-variance-like trade-off: too little
smoothing overfits noisy per-class variance estimates on correlated features, too much
smoothing flattens the class-conditional distributions until they stop being
informative.

## E. Failure / misconception

Own `GaussianNB` matches sklearn's exactly (100% prediction agreement on HAR), confirming
correctness - but at default settings it only reaches Macro-F1 0.768, well below the
Perceptron's 0.963. The failure is systematic: the biggest confusion is STANDING predicted
as SITTING (169/463 errors). This is the independence assumption breaking down - HAR's 561
features include many correlated signals (e.g. several axes of the same sensor), and
treating them as independent double-counts evidence, hurting exactly the class pair whose
distributions overlap most.

## F. Written-exam capsule

Naive Bayes is generative: it learns $P(X|Y)$ and $P(Y)$ separately, then uses Bayes'
rule to get $P(Y|X)$, rather than modelling $P(Y|X)$ directly like a discriminative
model. Its defining assumption is that features are conditionally independent given the
class, which makes the joint likelihood a simple product and training trivial, but the
assumption rarely holds exactly - when badly violated by correlated features, accuracy
suffers. Smoothing is needed because a single zero-probability or near-zero-variance
estimate can dominate the whole product and break the model on unseen data.

## G. Reflection

Before this week I could state Bayes' rule and the independence assumption, but hadn't
seen how much it costs on real, correlated features - the benchmark made the
Perceptron-vs-NaiveBayes gap (0.963 vs 0.768) concrete. The `var_smoothing` sweep was also
a surprise: not that it mattered, but that the default (`1e-9`) was near the worst choice
tested. Still uncertain: whether proper cross-validation would move the optimum much.

## H. Inquiry trail

Three episodes after pre-AI commit `6c9ac95`: (1) verified my NB code against
`tests/contract/` - 3/3 pass, reproducible; (2) got the benchmark script written (tooling)
for the `var_smoothing` sweep; (3) got a first draft of this post, grounded in (2)'s
numbers but not yet rewritten in my own words. Full rows: `AI_USE.md`, W06.

## Status

DRAFT - rewrite each section in your own words.
