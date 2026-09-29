# Pre-release catch-up (W01-W04) - Foundations + Decision Tree

## Foundations (Chapter 1)

Underfitting: the model is too simple to capture the pattern in the data, so high error on both the training set and the test set. 

Overfitting: the model is too complex and memorize noise in the training set, so low training error but high test error.

Bias is the error from a model being too simple to represent the true function (underfit). Variance is the error from a model being too sensitive to the particular training sample (overfit).
The expected squared prediction error decomposes as Bias^2 + Variance + irreducible noise σ^2. This is why "more complex is always better" is false. Past some point, more complexity gives lower bias but with more variance, so total expected error goes back up.
=> The practical response depends on failure mode: high bias needs more expressive models or better features, and high variance needs regularization, more data, or fewer features.

Model taxonomy: Supervised learning maps labeled input-output pairs. Unsupervised learning finds structure in unlabeled data through clustering, dimensionality reduction... Reinforcement learning learns a policy from reward through interaction.
Decision Trees are a non-parametric model, as they don't assume a fixed functional form beforehand, it learns from data.

Leakage check for HAR use case:
The UCI train and test sets are already split by subject, so no subject appears in both files. This prevents data leakage between subjects. If we create another validation split from the training data, it should also be done by subject, such as using GroupKFold with subject_train.txt, rather than randomly splitting individual rows. Otherwise, activity windows from the same subject could end up in both training and validation sets.

Any scaler, encoder, or feature selection should also be fitted only on the training data and then applied to the validation and test sets. Fitting these on the full dataset before splitting would allow information from the test set to leak into training.

Finally, the test set should remain untouched until the final comparison. All model tuning and decisions should be based only on the training and validation sets.

**Train/validation-curve diagnosis.** Ran a Decision Tree (`sklearn.tree.DecisionTreeClassifier`)
on the HAR training population with `max_depth` swept from 1 to 20, using a subject-aware
split (`GroupShuffleSplit` on `subject_train.txt`, seed 2452879) so no subject leaks
between the fit and validation portions. Script:
`experiments/part1_pre_midterm/foundations_learning_curve.py`; data:
`results/foundations_learning_curve.csv`; figure:
`results/figures/foundations_learning_curve.png`.

| max_depth | Train Macro-F1 | Val Macro-F1 |
|---|---|---|
| 1  | 0.231 | 0.224 |
| 3  | 0.713 | 0.688 |
| 8  | 0.981 | **0.858** (best) |
| 12 | 0.996 | 0.816 |
| 20 | 1.000 | 0.834 |

At `max_depth=1–2` both curves are low and close together - classic underfitting (the
tree is too shallow to separate 6 activities). From `max_depth≈8` onward, train Macro-F1
keeps climbing toward 1.0 while validation Macro-F1 stops improving and starts
oscillating - the train/val gap widens from ~0.12 at depth 8 to ~0.17 at depth 20, the
textbook overfitting signature. `max_depth=8` gives the best validation score observed
in this sweep and is the depth I'd pick as a starting capacity for the Decision Tree
catch-up build task, before any pruning.

## Decision Tree (Chapter 2)

Impurity measures how mixed the classes are within a node. Entropy is one common measure: H(S) = -sum_i p_i log2(p_i), where p_i is the fraction of samples belonging to class i. A pure node has entropy 0, while a binary node with p = 0.5 has entropy 1.

Information Gain measures how much a split reduces entropy: IG(S,A) = H(S) - sum_v (|S_v|/|S|) H(S_v). It is the parent's entropy minus the weighted average entropy of the child nodes. ID3 greedily chooses the attribute with the highest Information Gain at each node.

Worked example from the Play Tennis dataset: with 9 Yes and 5 No out of 14 samples, the initial entropy is approximately 0.94. If we split on Outlook, the three children are Sunny (2 Yes, 3 No), Overcast (4 Yes, 0 No), and Rain (3 Yes, 2 No). Their weighted average entropy is approximately 0.69, so IG(S, Outlook) ≈ 0.94 - 0.69 = 0.25. Since this is the highest Information Gain among the candidate attributes, Outlook is chosen as the root.

Continuous attributes require a different approach because there are infinitely many possible split values. ID3 does not directly handle numeric attributes. C4.5 sorts the values and considers thresholds at the midpoints between consecutive values, then evaluates each binary split A <= T versus A > T using Gain Ratio. The threshold with the best score is selected. CART also uses midpoint thresholds for continuous attributes, but evaluates them using Gini reduction for classification or SSE reduction for regression.

Missing values are handled differently across algorithms. C4.5 calculates the split quality using samples where the attribute is observed, then distributes samples with missing values fractionally across the child nodes based on how the observed samples were split. CART can use surrogate splits, where other features are ranked based on how well they reproduce the primary split. If the primary feature is missing at prediction time, the best available surrogate can be used. ID3 does not have built-in missing-value handling, so missing values need to be handled through preprocessing such as imputation or removal.

Pruning is used to reduce overfitting. A fully grown tree can create very small leaves that memorize noise in the training data. Pre-pruning stops the tree from growing when conditions such as minimum sample size, insufficient split improvement, or high node purity are reached. The node then predicts the majority class for classification or the mean target for regression.

Post-pruning grows a larger tree first and then removes branches that do not provide enough improvement. C4.5 uses an error-based pruning approach, comparing the estimated error of a subtree with the error of replacing it with a single leaf. CART uses cost-complexity pruning with R_alpha(T) = R(T) + alpha|T|, where the second term penalizes larger trees. It generates a sequence of smaller trees and typically uses cross-validation to select the appropriate tree size.

=> The main idea behind pruning is the same across these methods: a tree that perfectly fits the training data is not necessarily the tree that generalizes best.

## Release-day baseline diagnostic

Completed as a separate handwritten artifact: `exercises/release-baseline-w01-w02.pdf`
(date it honestly).

## Status

Foundations and Decision Tree sections rewritten in own words; train/validation-curve
diagnosis complete with real HAR results. Still outstanding: the handwritten baseline
diagnostic (`exercises/release-baseline-w01-w02.pdf`).
