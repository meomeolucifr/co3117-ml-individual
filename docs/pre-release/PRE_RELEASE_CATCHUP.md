# Pre-release catch-up (W01-W04) - Foundations + Decision Tree

## Foundations (Chapter 1)

Underfitting: the model is too simple to capture the pattern in the data, so both
training and test error are high. Overfitting: the model is too complex and memorizes
noise in the training set, so training error is low but test error is high.

Bias is the error from a model being too simple to represent the true function
(underfit). Variance is the error from a model being too sensitive to the particular
training sample (overfit). The expected squared prediction error decomposes as
Bias^2 + Variance + irreducible noise sigma^2, which is why "more complex is always
better" is false: past some point, extra complexity lowers bias but raises variance
more, so total expected error goes back up. High bias needs more expressive models or
better features; high variance needs regularization, more data, or fewer features.

Model taxonomy: supervised learning maps labeled input-output pairs; unsupervised
learning finds structure in unlabeled data (clustering, dimensionality reduction);
reinforcement learning learns a policy from reward through interaction. Decision Trees
are non-parametric - they don't assume a fixed functional form beforehand, the structure
is learned from data.

Leakage check for the HAR use case: the UCI train/test sets are already split by
subject, so no subject appears in both files, preventing leakage between subjects. Any
further validation split from the training data must also be subject-aware (e.g.
GroupKFold on subject_train.txt), never a random row split, or windows from the same
subject could end up in both train and validation. Any scaler, encoder, or feature
selection must be fit on training data only, then applied to validation/test - fitting
on the full dataset first would leak test-set information into training. The test set
stays untouched until the final comparison; all tuning uses only train/validation.

**Train/validation curve.** Ran a Decision Tree on the HAR training population,
max_depth 1-20, subject-aware split (GroupShuffleSplit on subject_train.txt, seed
2452879). Script: `experiments/part1_pre_midterm/foundations_learning_curve.py`;
results: `results/foundations_learning_curve.csv`, `results/figures/foundations_learning_curve.png`.

| max_depth | Train Macro-F1 | Val Macro-F1 |
|-|-|-|
| 1 | 0.231 | 0.224 |
| 3 | 0.713 | 0.688 |
| 8 | 0.981 | **0.858** |
| 12 | 0.996 | 0.816 |
| 20 | 1.000 | 0.834 |

At depth 1-2 both scores are low and close - underfitting, the tree is too shallow to
separate the 6 activities. Validation Macro-F1 peaks at depth 8 (0.858); after that
train Macro-F1 keeps climbing toward 1.0 while validation stops improving and
fluctuates, and the train-val gap widens from ~0.12 to ~0.17 by depth 20 - classic
overfitting. `max_depth=8` is a reasonable starting point for the Decision Tree build.

## Decision Tree (Chapter 2)

Entropy measures how mixed the classes in a node are: H(S) = -sum_i p_i log2(p_i). A
pure node has entropy 0; a balanced binary node has entropy 1. Information Gain is how
much a split reduces entropy: IG(S,A) = H(S) - sum_v (|S_v|/|S|) H(S_v), the parent's
entropy minus the weighted average entropy of the children. ID3 greedily picks the
attribute with the highest IG at each node.

Worked example (Play Tennis, 9 Yes/5 No of 14): H(S)~=0.94. Splitting on Outlook gives
Sunny (2Y/3N), Overcast (4Y/0N), Rain (3Y/2N), weighted entropy ~=0.69, so
IG(S,Outlook)~=0.25 - the highest among the candidates, so Outlook is the root.

Continuous attributes: ID3 can't split on numeric attributes directly. C4.5 sorts the
values, tries thresholds at midpoints between consecutive values, and scores each binary
split by Gain Ratio. CART does the same but scores by Gini reduction (classification) or
SSE reduction (regression).

Missing values: C4.5 computes split quality from samples with an observed value, then
distributes missing-value samples fractionally across children by the observed split
ratio. CART learns surrogate splits - other features ranked by how well they reproduce
the primary split - and falls back to the best surrogate when the primary is missing.
ID3 has no built-in handling; needs preprocessing.

Pruning reduces overfitting from tiny leaves that memorize noise. Pre-pruning stops
growth early (minimum sample size, negligible split improvement, high purity) and
predicts majority class/mean target. Post-pruning grows the full tree first, then
removes branches that don't improve enough: C4.5 compares a subtree's estimated error to
replacing it with one leaf; CART uses cost-complexity pruning,
R_alpha(T) = R(T) + alpha|T|, generating a sequence of smaller trees and picking alpha
by cross-validation. Either way, the tree that fits training data perfectly is not
necessarily the one that generalizes best.

## Release-day baseline diagnostic

Handwritten artifact: `exercises/release-baseline-w01-w02.pdf`.
