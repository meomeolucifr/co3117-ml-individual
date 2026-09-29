# AI_USE.md - mandatory AI-assisted inquiry-based learning (IBL) log

Every AI-assisted episode must be recorded here per the CO3117 IBL protocol (Section 12).
AI may only be used **after** a first-attempt commit/handwritten drill. Do not paste full
generated solutions - summarize the hint/question and what was independently verified.

## Template (one entry per episode)

```
### W__ / YYYY-MM-DD

- **Learning question:**
- **Pre-AI evidence:** <commit hash or handwritten first-attempt artifact path>
- **AI tool:**
- **Prompt purpose:** (Socratic hint / counterexample / debugging / quiz / ...)
- **Hint/question received (summary, not pasted solution):**
- **Verification source:** (CO3117 notes / textbook section / NPTEL lecture / repo)
- **What changed:** (misconception, derivation, test, or code decision corrected)
- **Closed-book reproduction:** Yes / Not yet - <link to delayed-retrieval artifact>
```

---

(Entries below, oldest first.)

### R0 / 2026-09-29

- **Learning question:** Are my ten handwritten answers on the release-day baseline diagnostic (Foundations: overfitting/underfitting, bias-variance; Decision Tree: entropy, Information Gain, continuous/missing-value handling, pruning) correct?
- **Pre-AI evidence:** Handwritten first attempt, no notes/AI while writing, scanned at exercises/release-baseline-w01-w02.pdf.
- **AI tool:** Claude Code.
- **Prompt purpose:** Verification/checking.
- **Hint/question received (summary):** Asked to check the 10 answers against course material.
- **Verification source:** CO3117 lecture slides "ML Introduction" and "Decision Trees"
- **What changed:** No corrections needed - all 10 answers confirmed correct on independent recomputation.
- **Closed-book reproduction:** Not yet.

### R0 / 2026-09-29

- **Learning question:** How do I empirically demonstrate the underfitting/overfitting bias-variance discussion from the catch-up post on the actual HAR dataset, instead of leaving it as a claim?
- **Pre-AI evidence:** N/A - this is Chapter 1 "Analysis" tier evidence, not a Depth-A from-scratch implementation, so it isn't gated behind a pre-AI commit the way the Perceptron build was.
- **AI tool:** Claude Code.
- **Prompt purpose:** Wrote the experiment script that sweeps DecisionTreeClassifier (max_depth = 1..20) on a subject-aware split of the HAR training set and records train/validation Macro-F1.
- **Hint/question received (summary):** N/A - full script provided directly since this is supporting/tooling code, not a required from-scratch component.
- **Verification source:** Ran the script myself and read the printed output/CSV directly. Confirmed the subject-disjoint assertion inside the script passes.
- **What changed:** Replaced the catch-up post's TODO placeholder with a real result table (best validation Macro-F1 = 0.858 at max_depth=8) and my own interpretation of where under/overfitting occurs.
- **Closed-book reproduction:** Not yet.

### W05 / 2026-09-29

- **Learning question:** Does ML-From-Scratch's Perceptron class implement the same update rule I derived and coded, and if not, what is it actually doing?
- **Pre-AI evidence:** Handwritten drill and pre-reference Perceptron / OneVsRestPerceptron implementation, both written before consulting the reference code.
- **AI tool:** Claude Code.
- **Prompt purpose:** Asked it to map the pasted ML-From-Scratch source to the theory from the drill and identify where it diverges from my implementation.
- **Hint/question received (summary):** Identified that the reference class runs full-batch gradient descent with a configurable activation/loss (Sigmoid + SquareLoss) rather than the discrete error-driven update, i.e. it is architecturally closer to the Delta rule than the classic Perceptron rule, despite the class name.
- **Verification source:** Read the pasted mlfromscratch source directly, line by line.
- **What changed:** Corrected the assumption that a shared class name implies a shared algorithm. No code changes needed because my own Perceptron already implements the correct rule.
- **Closed-book reproduction:** Not yet.
