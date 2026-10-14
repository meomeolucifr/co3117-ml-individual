# W07 - Genetic Algorithm + Bayesian Network / TAN (Part I closeout)

> DRAFT - rewrite in your own words before this counts as submission evidence
> (Section 12.3). Grounded in real results from this week's work.

## A. Concept capsule

A Genetic Algorithm (GA) is a population-based search heuristic: candidate solutions are
chromosomes, a fitness function scores each one, and a new generation is produced by
selecting fitter individuals and applying crossover and mutation. Unlike gradient descent
it needs no differentiability, only an evaluable fitness - suitable for feature selection,
where "which features" isn't a continuous parameter.

Separately: a Bayesian Network (BN) factorizes a joint as a product of local conditionals
over a DAG, $P(X_1,...,X_n) = \prod_i P(X_i \mid \text{parents}(X_i))$. d-separation reads
conditional independence off the graph: a chain/fork is blocked when the middle node is
observed; a collider is the opposite - blocked by default, active once conditioned on
("explaining away").

## B. One derivation or worked example

Chromosome design: 561-bit binary, one bit per feature, decoded as indices where the bit
is 1. An all-zero chromosome is invalid; repaired by forcing one random bit to 1 at init
and after mutation.

d-separation (DAG A->C, B->C, C->D, B->E): A,B with no evidence - blocked (collider at
C, inactive) -> independent. A,B given C - collider conditioned on -> active/dependent.
A,D given C - chain through C blocked by the mediator -> d-separated. C,E given B -
fork through B blocked by the common cause -> blocked.

## C. Code-to-theory trace

- Chromosome representation: `src/from_scratch/genetic_algorithm.py`,
  `GeneticFeatureSelection._initialize_population`.
- Fitness (validation Macro-F1 minus feature-count penalty): `_fitness`.
- Selection (tournament), crossover (uniform), mutation (bit-flip, rate
  `1/n_features`), elitism, patience-based stopping: `_select_parent`, `_crossover`,
  `_mutate`, main loop in `fit`.

## D. One controlled experiment

Question: how much does the result vary across seeds, and does one run tell you whether the
search converged? Ran `GeneticFeatureSelection` (population=50, up to 100 generations,
patience=20) on the frozen HAR split, 3 seeds, else fixed.

| Seed | Best fitness | Features selected | Stopping reason |
|---|---|---|---|
| 2452879 | 0.9063 | 204 | max_generations |
| 1 | 0.9217 | 185 | max_generations |
| 2 | 0.9024 | 212 | max_generations |

Spread across seeds: 0.0194 - small but real, confirming one run isn't representative. More
notable: all three seeds ran the full 100 generations without ever triggering the
20-generation patience stop, so the reported fitness is a lower bound.

## E. Failure / misconception

Found and fixed a bug while testing: `_mutate` referenced `self.mutation_rate` (raw
constructor arg, `None` by default) instead of `self.mutation_rate_` (resolved in `fit`,
defaulting to `1/n_features`). Since it stayed `None`, mutation crashed with
`TypeError: '<' not supported between 'float' and 'NoneType'`. Lesson: a resolved attribute
only helps if every method reads it.

## F. Written-exam capsule

A GA searches via selection, crossover, and mutation rather than gradients, suiting
discrete/non-differentiable problems like feature selection. A Bayesian Network factorizes
a joint over a DAG's local conditionals, and d-separation reads conditional independence
off the graph: chains/forks block dependence when conditioned on the middle node, while
colliders do the opposite - blocked by default, opened by conditioning on it or a
descendant.

## G. Reflection

Before this week I could describe GA operators abstractly but hadn't hit the concrete
resolved-vs-raw attribute bug - a detail "it runs" doesn't catch, since it only surfaced
once mutation actually executed. d-separation finally clicked as a mechanical procedure
(chain/fork: blocked when conditioned; collider: reverse) rather than memorization. Still
uncertain: whether the GA's feature sets converge to a similar subset across seeds or just
similar fitness via different subsets - addressed empirically in D, worth a closer look.

## H. Inquiry trail

Four episodes after pre-AI commit `f40df6c`: (1) a debugging question on the `_mutate`
`TypeError`, pointed to the exact line, no fix given, fixed myself; (2) the 3-seed experiment
script written/hardened (tooling), incl. resumable checkpointing after a memory crash; (3) a
first draft of this post incl. the BN/TAN capsule, not yet rewritten in my own words; (4) a
first draft of `report/part1_summary.pdf`, also not yet personalized. Rows: `AI_USE.md`, W07.

## Status

DRAFT - rewrite each section in your own words.
