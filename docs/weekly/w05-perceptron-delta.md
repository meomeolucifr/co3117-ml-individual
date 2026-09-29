# W05 - Perceptron and Delta rule

## A. Concept capsule

The perceptron is a single-layer linear binary classifier, computing z = w * x + b and predicts y_hat = sign(z) from an input x. Training is error-driven, where weights update only on a misclassified sample, via formula w <- w + n * y_i * x_i. Under the Perceptron Convergence Theorem, this halts in finite steps if the data is linearly separable. Therefore, if the characteristic of separable data is not met, the perceptron cannot converge. The Delta rule instead treats the output as continuous and derives its update from gradient descent on a differentiable loss like MSE, updating on every sample in proportion to the error magnitude. This is why it converges to a minimum-error solution on any dataset. The delta rule is also the mechanism that generalizes to backpropagation.

## B. One derivation or worked example

AND gate with bipolar encoding, n = 0.1, w_0=(0,0), b_0=0:

| Step | x | target | z=w*x+b | correct? | update |
|---|---|---|---|---|---|
| 1 | (1,1) | +1 | 0 | +1, correct | none |
| 2 | (1,-1) | -1 | 0 | +1, wrong | w += 0.1 * (-1) * (1, -1) = (-0.1, 0.1); b -= 0.1 |
| 3 | (-1,1) | -1 | 0.1 | +1, wrong | w += 0.1 * (-1) * (-1, 1) = (0, 0); b -= 0.1 |
| 4 | (-1,-1) | -1 | -0.2 | -1, correct | none |
| 5 | (1,1) | +1 | -0.2 | -1, wrong | w += 0.1 * (1) * (1, 1) = (0.1, 0.1); b += 0.1 |
| 6 | (1,-1) | -1 | -0.1 | -1, correct | none |
| 7 | (-1,1) | -1 | -0.1 | -1, correct | none |
| 8 | (-1,-1) | -1 | -0.3 | -1, correct | none |
| 9 | (1,1) | +1 | 0.1 | +1, correct | none |
| 10 | (1,-1) | -1 | -0.1 | -1, correct | none |
| 11 | (-1,1) | -1 | -0.1 | -1, correct | none |
| 12 | (-1,-1) | -1 | -0.3 | -1, correct | none |


## C. Code-to-theory trace

- Perceptron update rule w <- w + n * y_i * x_i only on misclassification on line 26.
- Convergence / early-stopping condition on lines 33-35.
- One-vs-rest extension to 6 HAR classes from line 47.

## D. One controlled experiment

Question: does my Perceptron still converge as a linearly-separable training set is a little bit adjusted by flipping 1 label? 
Controlled variable: number of flipped labels in an linearly-separable synthetic set (two Gaussian blobs, 120 samples, seed 2452879). Everything else (learning rate=0.1, epoch cap=200) was fixed.
Script at experiments/part1_pre_midterm/w05_perceptron_convergence.py

| Flipped labels | Converged? | Epochs run | Train accuracy |
|---|---|---|---|
| 0  | Yes | 2   | 1.000 |
| 1  | No  | 200 | 0.992 |
| 2  | No  | 200 | 0.983 |
| 3  | No  | 200 | 0.983 |
| 5  | No  | 200 | 0.958 |
| 8  | No  | 200 | 0.850 |
| 12 | No  | 200 | 0.900 |
| 20 | No  | 200 | 0.833 |

Result: the clean set converges almost immediately, after 2 epochs. The moment even a single label is flipped, the set is no longer linearly separable and converged never becomes True again for any noise level tested. It always runs the full 200-epoch cap, matching the oscillates forever prediction from drill question 3. However, the accuracy stays high at low noise despite never converging, and is not monotonic in the noise level because the perceptron doesn't optimize accuracy, it just stops wherever it happens to be in its oscillation cycle when the epoch cap hits. Figure is at results/figures/w05_perceptron_convergence.png.

## E. Failure / misconception

I assumed the reference Perceptron in ML-From-Scratch would implement the same error-correction update rule I did for this course. But in fact it doesn't. It has the Sigmoid activation and SquareLoss, which is a gradient-descent learner (Delta rule). The update rule in the code reference code allows it to learn the XOR function. But XOR is not linearly separable (as drawn in my w05-first-attempt.pdf), so no single perceptron can solve it without a hidden layer.

## F. Written-exam capsule

The perceptron learns a linear decision boundary by adjusting weights only when it misclassifies a training example, shifting the boundary toward the misclassified point's correct side. It is proved to converges in finite steps if and only if the two classes are linearly separable. On non-separable data like XOR, it oscillates forever. The Delta rule replaces this error-driven update with a continuous gradient-descent update derived from a differentiable loss like MSE, updating on every example by an amount proportional to the error, ultimately forming the minimum-error solution.

## G. Reflection

Before this week I could state the perceptron update formula but not explain why
the reference class Perceptron behaves so differently than mine. Now I can trace a
gradient-descent update back to the chain rule (loss.gradient * activation.gradient)
and connect it to the Delta rule instead of assuming a shared name implies a shared
algorithm. The thing that is still uncertain is: how sensitive the one-vs-rest scheme's tie-breaking is on HAR classes rather than the toy set. Need testing.

## H. Inquiry trail (mandatory whenever AI is used)

- Question: does ML-From-Scratch's Perceptron implement the same update rule as mine, and if not, what is it actually doing?
- Pre-AI evidence: handwritten drill and own pre-reference Perceptron implementation, both written before consulting the reference code.
- AI tool: Claude Code
- Prompt purpose: asked it to map the ML-From-Scratch code to the theory from the drill and identify where it differs from my implementation.
- Hint/question received: identified that the reference class from ML-From-Scratch runs full-batch gradient descent with Sigmoid activation function and SquareLoss rather than the error-driven update that I implemented.
- Verification source: cross-checked the claim by reading the mlfromscratch source directly line-by-line.
- What changed: corrected the assumption that same class name means same algorithm. I changed no code since my own Perceptron already implements the correct update rule.
- Closed-book reproduction: Rechecked myself the differences between my code and the one from ML-From-Scratch. In the reference code, it stated in the init function that it uses activation_function=Sigmoid, loss=SquareLoss. and also it updates weight differently:
self.W  -= self.learning_rate * grad_wrt_w
self.w0 -= self.learning_rate  * grad_wrt_w0

## Status
DONE