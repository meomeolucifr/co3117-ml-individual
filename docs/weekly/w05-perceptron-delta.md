# W05 - Perceptron and Delta rule

## A. Concept capsule

The perceptron is a single-layer linear binary classifier: z = w*x + b, y_hat =
sign(z). Training is error-driven - weights update only on a misclassified sample, via
w <- w + n*y_i*x_i. Under the Perceptron Convergence Theorem this halts in finite steps
if the data is linearly separable; otherwise it never converges. The Delta rule instead
treats the output as continuous and derives its update from gradient descent on a
differentiable loss like MSE, updating on every sample in proportion to the error
magnitude, so it converges to a minimum-error solution on any dataset. The Delta rule is
also the mechanism that generalizes to backpropagation.

## B. One derivation or worked example

AND gate, bipolar encoding, n=0.1, w_0=(0,0), b_0=0:

| Step | x | target | z | correct? | update |
|---|---|---|---|---|---|
| 1 | (1,1) | +1 | 0 | +1, correct | none |
| 2 | (1,-1) | -1 | 0 | +1, wrong | w += 0.1*(-1)*(1,-1) = (-0.1,0.1); b -= 0.1 |
| 3 | (-1,1) | -1 | 0.1 | +1, wrong | w += 0.1*(-1)*(-1,1) = (0,0); b -= 0.1 |
| 4 | (-1,-1) | -1 | -0.2 | -1, correct | none |

End of epoch 1: w=(0,0), b=-0.2. Epoch 2 has one more wrong case ((1,1), fixed once),
then every remaining pass is correct (verified through step 12), so the perceptron
converges on this separable toy set.

## C. Code-to-theory trace

- Perceptron update rule `w <- w + n*y_i*x_i`, only on misclassification:
  `src/from_scratch/perceptron.py`, `perceptron_update`.
- Convergence / early-stopping: `Perceptron.fit`.
- One-vs-rest extension to 6 HAR classes: `OneVsRestPerceptron`.

## D. One controlled experiment

Question: does the perceptron still converge once a linearly-separable set is nudged
away from separability by flipping labels? Controlled variable: number of flipped
labels in a synthetic 2-blob set (120 samples, seed 2452879); lr=0.1, epoch cap=200
fixed. Script: `experiments/part1_pre_midterm/w05_perceptron_convergence.py`.

| Flipped | Converged? | Epochs | Train acc |
|---|---|---|---|
| 0 | Yes | 2 | 1.000 |
| 1 | No | 200 | 0.992 |
| 8 | No | 200 | 0.850 |
| 20 | No | 200 | 0.833 |

Result: the clean set converges in 2 epochs; one flipped label is enough to make
`converged` stay False for every noise level tested, matching the "oscillates forever"
prediction from drill Q3. Accuracy stays high at low noise but is non-monotonic, since
the perceptron just stops wherever it is in its oscillation cycle at the epoch cap.

## E. Failure / misconception

I assumed ML-From-Scratch's `Perceptron` implements the same error-correction rule I
did. It doesn't: it uses Sigmoid activation and SquareLoss, i.e. a gradient-descent
(Delta rule) learner. Separately: XOR is not linearly separable (bipolar points sit on
opposite diagonal corners of the unit square), so no single perceptron of either kind
can solve it without a hidden layer.

## F. Written-exam capsule

The perceptron adjusts weights only on misclassification, shifting the boundary toward
the correct side; it converges in finite steps iff the data is linearly separable, else
it oscillates forever. The Delta rule replaces this with a continuous gradient-descent
update proportional to the error on every example, converging to a minimum-error
solution even on non-separable data.

## G. Reflection

Before this week I could state the perceptron update formula but not explain why
ML-From-Scratch's same-named class behaves so differently. Now I can connect that
gradient-descent update to the Delta rule instead of assuming a shared name means a
shared algorithm. Still uncertain: how sensitive one-vs-rest tie-breaking is on real,
overlapping HAR classes rather than a toy set.

## H. Inquiry trail (mandatory whenever AI is used)

- Question: does ML-From-Scratch's Perceptron implement the same update rule as mine?
- Pre-AI evidence: handwritten drill and own pre-reference Perceptron implementation,
  both written before consulting the reference code.
- AI tool: Claude Code.
- Prompt purpose: map the ML-From-Scratch code to the drill theory, find divergences.
- Hint/question received: the reference class runs full-batch gradient descent with
  Sigmoid activation and SquareLoss rather than the error-driven update I implemented.
- Verification source: read the mlfromscratch source directly, line by line.
- What changed: corrected the assumption that same class name means same algorithm; no
  code change needed, my own Perceptron already implements the correct rule.
- Closed-book reproduction: Not yet.

## Status

DONE
