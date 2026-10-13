# W06 drill corrections

Self-check of `exercises/w06-first-attempt.pdf` (5 questions: Bayes rule, naive
conditional-independence assumption, smoothing, why NB is generative, a posterior
calculation) after the first-attempt commit.

## What was checked

- **A (Bayes rule):** $P(Y|X) = P(X|Y)P(Y)/P(X)$, correct definition of each term
  (prior, likelihood, posterior, evidence).
- **B (naive independence):** $P(X_1,...,X_n|Y) = \prod_i P(X_i|Y)$, correctly stated
  as "no additional info about one feature once the class is known."
- **C (smoothing):** correctly identifies that an unseen feature value gives a zero
  conditional probability, which zeroes the whole product and the class posterior;
  add-one smoothing adds 1 to each count and adjusts the denominator to avoid this.
- **D (generative):** correctly explains NB models $P(X,Y) = P(X|Y)P(Y)$ then applies
  Bayes' rule, contrasted with discriminative models (e.g. logistic regression) that
  model $P(Y|X)$ directly without modelling how features are generated.
- **E (numeric):** $P(X{=}1) = 0.8(0.3) + 0.2(0.7) = 0.38$,
  $P(Y{=}1|X{=}1) = 0.8(0.3)/0.38 \approx 0.6316$. Recomputed independently and matches.

## Result

All five answers correct - no conceptual or computational errors found.

## Verification source

CO3117 course material on Bayes' rule, Naive Bayes, and generative vs. discriminative
models; recomputed the posterior in E independently rather than just checking the
arithmetic as written.
