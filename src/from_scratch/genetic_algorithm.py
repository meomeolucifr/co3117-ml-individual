import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score


class GeneticFeatureSelection:
    def __init__(
        self,
        population_size=50,
        n_generations=100,
        crossover_rate=0.8,
        mutation_rate=None,
        tournament_size=3,
        penalty=0.05,
        patience=20,
        seed=None,
        model_factory=None,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.penalty = penalty
        self.patience = patience
        self.seed = seed
        self.model_factory = model_factory or GaussianNB

    def _fitness(self, chromosome, X_train, y_train, X_val, y_val):
        selected = np.flatnonzero(chromosome)

        # A chromosome must select at least one feature
        if len(selected) == 0:
            return -np.inf

        model = self.model_factory()
        model.fit(X_train[:, selected], y_train)
        predictions = model.predict(X_val[:, selected])

        score = f1_score(
            y_val, predictions, average="macro", zero_division=0
        )
        feature_ratio = len(selected) / X_train.shape[1]

        return score - self.penalty * feature_ratio

    def _initialize_population(self, n_features, rng):
        population = rng.integers(
            0, 2, size=(self.population_size, n_features)
        ).astype(np.int8)

        # Ensure every chromosome selects at least one feature
        for chromosome in population:
            if not chromosome.any():
                chromosome[rng.integers(n_features)] = 1

        return population

    def _select_parent(self, population, fitness, rng):
        candidates = rng.choice(
            len(population),
            size=min(self.tournament_size, len(population)),
            replace=False,
        )
        winner = candidates[np.argmax(fitness[candidates])]
        return population[winner].copy()

    def _crossover(self, parent1, parent2, rng):
        if rng.random() >= self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # Uniform crossover
        mask = rng.random(len(parent1)) < 0.5

        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)

        return child1, child2

    def _mutate(self, chromosome, rng):
        mask = rng.random(len(chromosome)) < self.mutation_rate_
        chromosome[mask] = 1 - chromosome[mask]

        # Repair empty chromosome
        if not chromosome.any():
            chromosome[rng.integers(len(chromosome))] = 1

        return chromosome

    def fit(self, X_train, y_train, X_val, y_val):
        X_train = np.asarray(X_train)
        X_val = np.asarray(X_val)
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)

        if X_train.ndim != 2 or X_val.ndim != 2:
            raise ValueError("X_train and X_val must be 2D arrays")
        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError("Train and validation feature counts differ")
        if len(X_train) != len(y_train) or len(X_val) != len(y_val):
            raise ValueError("X and y lengths do not match")
        if X_train.shape[1] == 0:
            raise ValueError("At least one feature is required")
        if self.population_size < 2 or self.n_generations < 1:
            raise ValueError("Invalid population size or generation count")
        if self.tournament_size < 1 or self.patience < 1:
            raise ValueError("tournament_size and patience must be positive")

        rng = np.random.default_rng(self.seed)
        n_features = X_train.shape[1]

        # Standard bit-flip mutation rate: 1 / number of features
        mutation_rate = (
            1.0 / n_features
            if self.mutation_rate is None
            else self.mutation_rate
        )
        if not 0 <= mutation_rate <= 1:
            raise ValueError("mutation_rate must be in [0, 1]")
        self.mutation_rate_ = mutation_rate

        population = self._initialize_population(n_features, rng)

        fitness = np.array([
            self._fitness(ch, X_train, y_train, X_val, y_val)
            for ch in population
        ])

        best_idx = np.argmax(fitness)
        self.best_chromosome_ = population[best_idx].copy()
        self.best_fitness_ = float(fitness[best_idx])
        self.best_generation_ = 0
        self.history_ = []
        self.n_generations_run_ = 0

        generations_without_improvement = 0

        for generation in range(1, self.n_generations + 1):
            # Elitism: preserve the best chromosome
            elite_idx = np.argmax(fitness)
            new_population = [population[elite_idx].copy()]

            while len(new_population) < self.population_size:
                parent1 = self._select_parent(population, fitness, rng)
                parent2 = self._select_parent(population, fitness, rng)

                child1, child2 = self._crossover(parent1, parent2, rng)
                child1 = self._mutate(child1, rng)
                child2 = self._mutate(child2, rng)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = np.asarray(new_population, dtype=np.int8)

            fitness = np.array([
                self._fitness(ch, X_train, y_train, X_val, y_val)
                for ch in population
            ])

            generation_best_idx = np.argmax(fitness)
            generation_best_fitness = float(fitness[generation_best_idx])

            if generation_best_fitness > self.best_fitness_:
                self.best_fitness_ = generation_best_fitness
                self.best_chromosome_ = (
                    population[generation_best_idx].copy()
                )
                self.best_generation_ = generation
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            self.history_.append({
                "generation": generation,
                "best_fitness": self.best_fitness_,
                "generation_best_fitness": generation_best_fitness,
                "selected_features": int(
                    self.best_chromosome_.sum()
                ),
            })
            self.n_generations_run_ = generation

            if generations_without_improvement >= self.patience:
                self.stopping_reason_ = "patience"
                break
        else:
            self.stopping_reason_ = "max_generations"

        self.selected_features_ = np.flatnonzero(
            self.best_chromosome_
        )
        self.n_features_selected_ = len(self.selected_features_)

        return self

    def transform(self, X):
        return np.asarray(X)[:, self.selected_features_]

    def fit_transform(self, X_train, y_train, X_val, y_val):
        self.fit(X_train, y_train, X_val, y_val)
        return self.transform(X_train), self.transform(X_val)
