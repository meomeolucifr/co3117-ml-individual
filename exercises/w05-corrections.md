# W05 drill corrections

Self-check of `exercises/w05-first-attempt.pdf` (4 questions: perceptron update rule,
linear separability, why XOR fails, perceptron vs delta rule) against CO3117 course
material, after the first-attempt commit.

## What was checked

- **Q1 (perceptron update rule):** formula $\mathbf{w}\leftarrow\mathbf{w}+\eta y_i\mathbf{x}_i$,
  geometric explanation (rotating the decision hyperplane toward misclassified points),
  Perceptron Convergence Theorem and its linear-separability precondition.
- **Q2 (linear separability):** definition via a separating hyperplane, link to why
  Perceptron fails to converge without it.
- **Q3 (XOR):** bipolar encoding of the 4 points, diagonal-corner layout, conclusion that
  no single line separates the classes.
- **Q4 (perceptron vs delta rule):** discrete/error-triggered update vs continuous
  gradient-descent update; MSE minimization; relation to backpropagation.

## Result

All four answers were correct on review - no conceptual or computational errors found.
One point noted for the written-exam capsule (not a correction, a deepening question):
be able to explain *why* the Delta rule needs a differentiable activation function while
the discrete Perceptron rule does not, since this is the kind of distinction the sample
final likes to probe.

## Verification source

CO3117 course material already covered for this topic (lecture content on Perceptron,
linear separability, and the Delta/Widrow-Hoff rule); cross-checked against the standard
XOR non-separability argument and the Perceptron Convergence Theorem statement.
