# W07 drill corrections

Self-check of `exercises/w07-first-attempt.pdf` (5 questions: chromosome
representation, fitness function, selection/crossover/mutation, premature
convergence, a mutation example) after the first-attempt commit.

## What was checked

- **A (chromosome representation):** correctly describes a binary chromosome where
  each bit encodes inclusion/exclusion of one feature, with a worked example (5-bit
  chromosome `10110` selecting features 1, 3, 4).
- **B (fitness function):** correctly describes fitness as evaluating solution quality
  and reproduction suitability, and correctly contrasts it with gradient descent's
  objective (GA doesn't need differentiability; it uses crossover/mutation/selection
  instead of gradients).
- **C (selection/crossover/mutation):** correctly distinguishes selection (choosing
  existing individuals by fitness) from crossover (combining genetic material from two
  parents) and mutation (randomly changing genes to introduce variation).
- **D (premature convergence):** correctly identifies it as losing population diversity
  and converging to a sub-optimal solution when exploitation dominates exploration;
  correct interventions (increase mutation, preserve diversity, reduce selection
  pressure).
- **E (mutation example):** `1010` with bit 2 flipped → `1110`. Recomputed
  independently and matches.

## Result

All five answers correct - no conceptual or computational errors found.

## Verification source

CO3117 course material on Genetic Algorithms (representation, fitness, operators,
exploration vs. exploitation); recomputed the mutation example in E independently.
