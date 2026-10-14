# W05: Perceptron and Delta Rule

## A. Concept

The Perceptron is a linear binary classifier that updates its weights only when it misclassifies a sample, using w = w + learning_rate * target * input. It converges in finite steps when data is linearly separable but may keep updating otherwise.

The Delta rule uses gradient descent to minimize a differentiable loss such as MSE. It updates weights according to prediction error and can minimize loss even on non-separable data, although perfect classification is not guaranteed. This principle also underlies backpropagation.

## B. Worked example

I tested a bipolar AND gate with learning rate 0.1, initial weights (0, 0), and bias 0.

| Step | Input | Target | Output z | Result | Update |
|---|---|---|---|---|---|
| 1 | (1,1) | +1 | 0 | Correct | None |
| 2 | (1,-1) | -1 | 0 | Wrong | w=(-0.1, 0.1), b=-0.1 |
| 3 | (-1,1) | -1 | 0.1 | Wrong | w=(0, 0), b=-0.2 |
| 4 | (-1,-1) | -1 | -0.2 | Correct | None |

The model made additional corrections in the second epoch and eventually converged on this linearly separable dataset.

## C. Code and theory

My standard update rule is implemented in src/from_scratch/perceptron.py, while Perceptron.fit handles training and convergence checks. OneVsRestPerceptron extends the classifier to six HAR classes.

I initially assumed the ML-From-Scratch Perceptron used the same rule. After reading its source, I found it uses Sigmoid activation, SquareLoss, and gradient descent, making it closer to the Delta rule.

## D. Experiment

I tested convergence on 120 synthetic two-blob samples using seed 2452879, learning rate 0.1, and a 200-epoch limit. I varied the number of flipped labels.

| Flipped labels | Converged? | Epochs | Accuracy |
|---:|---|---:|---:|
| 0 | Yes | 2 | 1.000 |
| 1 | No | 200 | 0.992 |
| 8 | No | 200 | 0.850 |
| 20 | No | 200 | 0.833 |

The clean dataset converged in two epochs, while every noisy setting reached the epoch limit without convergence. Even one flipped label left accuracy at 99.2%, showing that high accuracy does not necessarily mean convergence. These results apply to this dataset and setup.

## E. Mistakes and misconceptions

I learned that identical class names do not guarantee identical algorithms. My implementation already used the standard Perceptron rule, so no changes were needed. I also reviewed why a single Perceptron cannot solve XOR: its classes cannot be separated by one linear boundary, regardless of the learning rule. A hidden layer is needed for a more complex decision boundary.

## F. Written exam summary

The Perceptron updates only on misclassified samples and converges when data is linearly separable. The Delta rule uses gradient descent to minimize prediction loss and can handle non-separable data, though errors may remain.

## G. Reflection

This week, I learned to distinguish algorithms by their actual update rules rather than their names. I still want to investigate how tie-breaking affects OneVsRestPerceptron on overlapping HAR classes.

## H. AI inquiry

I used Claude Code to compare my implementation with ML-From-Scratch after completing my own drill and code. I verified its explanation by reading the source line by line. No code changes were needed. I have not yet reproduced the derivations from memory, so a closed-book review remains.

## Status

DONE