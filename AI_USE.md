# AI_USE

One row per AI-assisted learning episode (assignment Section 12.5). Weeks with a row here must
also have section H (Inquiry trail) in that week's post. Pre-AI evidence is a commit hash from W05 on;
for the W01-W04 catch-up cite the release-day baseline file.

| Week/date | Learning question | Pre-AI evidence | AI tool | Prompt purpose | Hint/question received | Verification source | What changed | Closed-book reproduction |
|---|---|---|---|---|---|---|---|---|
<!-- Example row (a real row starts with the week, like this): | W05 / 2026-10-09 | Why does Laplace smoothing fix zero counts? | 3f9c2ab | (tool) | Socratic hint | Asked me to compute one count by hand | Murphy 3.3.4 | Fixed my smoothing denominator | Yes, see exercises/w05-retrieval.md | -->
| W05 / 2026-09-29 | Are my 10 handwritten baseline-diagnostic answers (Foundations, Decision Tree) correct? | 325b9bb | Claude Code | Verification/checking | Asked to check the 10 answers against course material | CO3117 slides: ML Introduction, Decision Trees | No corrections needed, all 10 confirmed correct | Not yet |
| W05 / 2026-09-29 | How do I empirically show the under/overfitting discussion from the catch-up post on real HAR data? | 325b9bb | Claude Code | Wrote the capacity-sweep experiment script (tooling, not a Depth-A component) | Full script for DecisionTreeClassifier max_depth=1..20 on a subject-aware split | Ran it myself, read the printed output and CSV | Replaced the TODO with a real result table (best val Macro-F1 0.858 at depth 8) | Not yet |
| W05 / 2026-09-29 | Does ML-From-Scratch's Perceptron implement the same update rule I coded? | 4eaf678 | Claude Code | Map pasted reference code to my drill theory, find divergence | Reference runs full-batch GD with Sigmoid+SquareLoss (Delta rule), not the discrete Perceptron rule | Read the pasted mlfromscratch source line by line | Corrected the assumption that same class name means same algorithm; no code change needed | Not yet |
