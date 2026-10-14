# W06: Naive Bayes

## A. Concept

Naive Bayes is a generative classifier that learns class probabilities and feature distributions, then applies Bayes' rule to predict the class of a sample. It assumes features are conditionally independent given the class, simplifying probability calculations.

GaussianNB handles continuous features by estimating the mean and variance of each feature for each class. Its var_smoothing parameter helps prevent numerical problems caused by very small variances.

## B. Worked example

Given P(Y=1) = 0.3, P(X=1|Y=1) = 0.8, and P(X=1|Y=0) = 0.2:

P(X=1) = 0.8 * 0.3 + 0.2 * 0.7 = 0.38

P(Y=1|X=1) = (0.8 * 0.3) / 0.38 = 0.632

Therefore, the probability of class 1 given X=1 is approximately 63.2%.

## C. Code and theory

In src/from_scratch/naive_bayes.py, GaussianNB.fit estimates per-class feature means and variances, stored in self.theta_ and self.var_. The predict_log_proba method sums Gaussian log-likelihoods across features and uses log-sum-exp normalization to obtain stable log probabilities.

## D. Experiment

I tested five var_smoothing values on HAR using a fixed subject-aware GroupShuffleSplit with seed 2452879.

| var_smoothing | Macro-F1 |
|---|---:|
| 1e-9 (default) | 0.7679 |
| 1e-6 | 0.8100 |
| 1e-3 | 0.8411 |
| 1e-1 | 0.8172 |
| 1.0 | 0.8077 |

The best result was 0.8411 at 1e-3, improving Macro-F1 by 0.0732 over the default. Performance declined at higher values, suggesting that too much smoothing reduces the differences between class distributions. Further validation is needed to determine whether this setting generalizes to other splits.

## E. Mistakes and misconceptions

My GaussianNB implementation matched sklearn's predictions on HAR 100%, but its default Macro-F1 was only 0.768, compared with 0.963 for the Perceptron. One major confusion involved STANDING being predicted as SITTING, with 169 errors out of 463 STANDING samples.

HAR contains many correlated features. Treating them as independent can count related evidence multiple times, potentially hurting classification when activities have overlapping sensor patterns. This showed me that correct implementation does not necessarily mean strong performance.

## F. Written exam summary

Naive Bayes estimates P(Y) and P(X|Y), then uses Bayes' rule to calculate P(Y|X). Its conditional independence assumption simplifies learning but may be unrealistic for correlated features. GaussianNB estimates per-class means and variances, while variance smoothing helps stabilize calculations.

## G. Reflection

The experiment made the impact of the independence assumption more concrete. I was also surprised that the default var_smoothing performed worst among the tested settings. I still want to check whether cross-validation would change the best value, since the current results come from one fixed split.

## H. AI inquiry

After pre-AI commit 6c9ac95, I used AI to check my implementation, with all three contract tests passing, prepare the var_smoothing benchmark script, and draft this summary. I verified the implementation and experimental results, but the initial draft still needed rewriting in my own words. The full record is in AI_USE.md under W06.

## Status

DONE