# W07: Genetic Algorithm and Bayesian Network / TAN

## A. Concept

A Genetic Algorithm (GA) is a search method that evolves a population of candidate solutions using selection, crossover, and mutation. It does not require gradients, making it suitable for discrete problems such as feature selection.

A Bayesian Network (BN) represents variables as a directed acyclic graph (DAG), where the joint probability is the product of each variable's conditional probability given its parents. D-separation identifies conditional independence: observing the middle node blocks a chain or fork, while observing a collider or its descendant can activate a previously blocked path.

## B. Worked example

For HAR feature selection, each chromosome contains 561 bits, with 1 meaning a feature is selected and 0 meaning it is excluded. To avoid invalid solutions, I ensured that every chromosome contained at least one selected feature.

For a DAG with edges A -> C, B -> C, C -> D, and B -> E, A and B are independent by default because C is a collider. Observing C activates their path, making them potentially dependent. A and D become independent given C because it blocks the chain, while C and E become independent given B because B is their common cause.

## C. Code and theory

In src/from_scratch/genetic_algorithm.py, _initialize_population creates chromosomes and _fitness evaluates validation Macro-F1 minus a penalty for selecting too many features. The algorithm uses tournament selection, uniform crossover, bit-flip mutation, elitism, and patience-based stopping. The mutation rate defaults to 1 divided by the number of features.

## D. Experiment

I tested how results varied across three random seeds on the same HAR split, using a population of 50, a maximum of 100 generations, and patience of 20.

| Seed | Best fitness | Features selected | Stopping reason |
|---|---:|---:|---|
| 2452879 | 0.9063 | 204 | max_generations |
| 1 | 0.9217 | 185 | max_generations |
| 2 | 0.9024 | 212 | max_generations |

Fitness varied from 0.9024 to 0.9217. All runs reached the generation limit without triggering early stopping. This suggests that results depend somewhat on the random seed, and reaching the maximum number of generations does not necessarily mean the search has converged. I still need to compare the actual feature subsets to see whether different runs select similar features.

## E. Mistakes and misconceptions

I found a bug in _mutate, which used self.mutation_rate instead of self.mutation_rate_. The first attribute was None by default, while the second stored the resolved mutation rate. This caused a TypeError during mutation. I fixed the issue by using the resolved attribute. The experience taught me to check how parameters are used throughout the implementation, not just how they are initialized.

## F. Written exam summary

A GA searches through selection, crossover, and mutation, making it useful for discrete optimization. A BN factorizes a joint probability distribution using local conditional probabilities. D-separation identifies conditional independence by checking paths in the graph. Conditioning blocks chains and forks but can activate colliders.

## G. Reflection

This week, I learned that small implementation mistakes can break an algorithm even when its overall design is correct. Working through d-separation also helped me understand the rules instead of simply memorizing them. I am still unsure whether different GA seeds select similar feature subsets or achieve similar fitness using different features.

## H. AI inquiry

After pre-AI commit f40df6c, I used AI to investigate the mutation TypeError, prepare the three-seed experiment script with resumable checkpointing, draft this summary, and draft report/part1_summary.pdf. The debugging assistance pointed me to the relevant line without giving me the fix, which I implemented myself. The drafts still required revision in my own words. The full record is in AI_USE.md under W07.

## Status

DONE