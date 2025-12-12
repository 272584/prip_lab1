import numpy as np
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple
import time
import random


@dataclass
class City:
    x: float
    y: float

    def distance_to(self, other: 'City') -> float:
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class TSPProblem:
    def __init__(self, cities: List[City]):
        self.cities = cities
        self.n = len(cities)
        # Prekalkulacja macierzy odległości dla O(1) dostępu
        self.distance_matrix = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                self.distance_matrix[i][j] = cities[i].distance_to(cities[j])

    def evaluate_route(self, route: List[int]) -> float:
        total = 0.0
        for i in range(len(route) - 1):
            total += self.distance_matrix[route[i]][route[i + 1]]
        total += self.distance_matrix[route[-1]][route[0]]
        return total


class Individual:
    def __init__(self, genotype: List[int], fitness: float = None):
        self.genotype = genotype
        self.fitness = fitness

    def copy(self):
        return Individual(self.genotype.copy(), self.fitness)


# ============================================================================
# FUNKCJE GLOBALNE DO RÓWNOLEGŁEGO PRZETWARZANIA
# ============================================================================

def evaluate_batch(args):
    """Ewaluacja grupy osobników"""
    individuals, distance_matrix = args
    results = []
    for individual in individuals:
        if individual.fitness is None:
            route = individual.genotype
            total = 0.0
            for i in range(len(route) - 1):
                total += distance_matrix[route[i]][route[i + 1]]
            total += distance_matrix[route[-1]][route[0]]
            individual.fitness = total
        results.append(individual)
    return results


# ============================================================================
# WERSJA SEKWENCYJNA
# ============================================================================

class SequentialGeneticAlgorithm:
    def __init__(self, problem: TSPProblem,
                 population_size: int = 100,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.2,
                 tournament_size: int = 3):
        self.problem = problem
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize_population(self) -> List[Individual]:
        population = []

        for _ in range(self.population_size // 2):
            genotype = list(range(self.problem.n))
            random.shuffle(genotype)
            population.append(Individual(genotype))

        for _ in range(self.population_size - len(population)):
            genotype = self._nearest_neighbor_heuristic()
            for _ in range(random.randint(1, 5)):
                i, j = random.sample(range(len(genotype)), 2)
                genotype[i], genotype[j] = genotype[j], genotype[i]
            population.append(Individual(genotype))

        return population

    def _nearest_neighbor_heuristic(self) -> List[int]:
        start = random.randint(0, self.problem.n - 1)
        unvisited = set(range(self.problem.n))
        route = [start]
        unvisited.remove(start)

        while unvisited:
            current = route[-1]
            nearest = min(unvisited,
                          key=lambda city: self.problem.distance_matrix[current][city])
            route.append(nearest)
            unvisited.remove(nearest)

        return route

    def evaluate_individual(self, individual: Individual) -> Individual:
        if individual.fitness is None:
            individual.fitness = self.problem.evaluate_route(individual.genotype)
        return individual

    def evaluate_population(self, population: List[Individual]) -> List[Individual]:
        evaluated = []
        for individual in population:
            evaluated.append(self.evaluate_individual(individual))
        return evaluated

    def tournament_selection(self, population: List[Individual]) -> Individual:
        tournament = random.sample(population, self.tournament_size)
        return min(tournament, key=lambda ind: ind.fitness)

    def order_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        n = len(parent1.genotype)
        point1, point2 = sorted(random.sample(range(n), 2))

        def create_child(p1, p2):
            child = [-1] * n
            child[point1:point2] = p1.genotype[point1:point2]
            p2_filtered = [gene for gene in p2.genotype if gene not in child]
            idx = 0
            for i in range(n):
                if child[i] == -1:
                    child[i] = p2_filtered[idx]
                    idx += 1
            return child

        child1_genotype = create_child(parent1, parent2)
        child2_genotype = create_child(parent2, parent1)

        return Individual(child1_genotype), Individual(child2_genotype)

    def swap_mutation(self, individual: Individual) -> Individual:
        genotype = individual.genotype.copy()
        i, j = random.sample(range(len(genotype)), 2)
        genotype[i], genotype[j] = genotype[j], genotype[i]
        return Individual(genotype)

    def inversion_mutation(self, individual: Individual) -> Individual:
        genotype = individual.genotype.copy()
        i, j = sorted(random.sample(range(len(genotype)), 2))
        genotype[i:j + 1] = reversed(genotype[i:j + 1])
        return Individual(genotype)

    def crossover_population(self, population: List[Individual]) -> List[Individual]:
        offspring = []
        n_pairs = self.population_size // 2

        for _ in range(n_pairs):
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)

            while parent1.genotype == parent2.genotype:
                parent2 = self.tournament_selection(population)

            if random.random() < self.crossover_prob:
                child1, child2 = self.order_crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])

        return offspring[:self.population_size]

    def mutate_population(self, population: List[Individual]) -> List[Individual]:
        mutated = []

        for individual in population:
            if random.random() < self.mutation_prob:
                if random.random() < 0.5:
                    mutated.append(self.swap_mutation(individual))
                else:
                    mutated.append(self.inversion_mutation(individual))
            else:
                mutated.append(individual)

        return mutated

    def run(self, max_generations: int = 1000, time_limit: float = None,
            verbose: bool = True) -> Tuple[List[int], float]:
        start_time = time.time()

        population = self.initialize_population()
        population = self.evaluate_population(population)

        best_individual = min(population, key=lambda ind: ind.fitness)
        self.best_solution = best_individual.genotype
        self.best_fitness = best_individual.fitness

        for generation in range(max_generations):
            if time_limit and (time.time() - start_time) > time_limit:
                break

            offspring = self.crossover_population(population)
            offspring = self.mutate_population(offspring)
            offspring = self.evaluate_population(offspring)

            population = offspring
            current_best = min(population, key=lambda ind: ind.fitness)

            if current_best.fitness < self.best_fitness:
                self.best_fitness = current_best.fitness
                self.best_solution = current_best.genotype
                if verbose:
                    print(f"Generacja {generation}: Nowa najlepsza fitness = {self.best_fitness:.2f}")

            worst_idx = max(range(len(population)), key=lambda i: population[i].fitness)
            population[worst_idx] = Individual(self.best_solution, self.best_fitness)

        elapsed_time = time.time() - start_time
        if verbose:
            print(f"\nAlgorytm zakończony po {elapsed_time:.2f}s")
            print(f"Najlepsza fitness: {self.best_fitness:.2f}")

        return self.best_solution, self.best_fitness


# ============================================================================
# WERSJA RÓWNOLEGŁA Z PROCESAMI
# ============================================================================

class ParallelGeneticAlgorithm:
    def __init__(self, problem: TSPProblem,
                 population_size: int = 100,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.2,
                 tournament_size: int = 3,
                 n_workers: int = 4):
        self.problem = problem
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.n_workers = n_workers
        self.best_solution = None
        self.best_fitness = float('inf')

        # Utrzymujemy jeden pool przez cały czas działania
        self.executor = None

    def initialize_population(self) -> List[Individual]:
        population = []

        for _ in range(self.population_size // 2):
            genotype = list(range(self.problem.n))
            random.shuffle(genotype)
            population.append(Individual(genotype))

        for _ in range(self.population_size - len(population)):
            genotype = self._nearest_neighbor_heuristic()
            for _ in range(random.randint(1, 5)):
                i, j = random.sample(range(len(genotype)), 2)
                genotype[i], genotype[j] = genotype[j], genotype[i]
            population.append(Individual(genotype))

        return population

    def _nearest_neighbor_heuristic(self) -> List[int]:
        start = random.randint(0, self.problem.n - 1)
        unvisited = set(range(self.problem.n))
        route = [start]
        unvisited.remove(start)

        while unvisited:
            current = route[-1]
            nearest = min(unvisited,
                          key=lambda city: self.problem.distance_matrix[current][city])
            route.append(nearest)
            unvisited.remove(nearest)

        return route

    def evaluate_population_parallel(self, population: List[Individual]) -> List[Individual]:
        """Równoległa ewaluacja z wykorzystaniem ProcessPoolExecutor i batch processing"""

        # Podziel populację na batche dla każdego workera
        batch_size = len(population) // self.n_workers
        if batch_size == 0:
            batch_size = 1

        batches = []
        for i in range(0, len(population), batch_size):
            batch = population[i:i + batch_size]
            batches.append((batch, self.problem.distance_matrix))

        # Przetwarzaj batche równolegle
        results = list(self.executor.map(evaluate_batch, batches))

        # Spłaszcz wyniki
        evaluated = []
        for batch_result in results:
            evaluated.extend(batch_result)

        return evaluated

    def tournament_selection(self, population: List[Individual]) -> Individual:
        tournament = random.sample(population, self.tournament_size)
        return min(tournament, key=lambda ind: ind.fitness)

    def order_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        n = len(parent1.genotype)
        point1, point2 = sorted(random.sample(range(n), 2))

        def create_child(p1, p2):
            child = [-1] * n
            child[point1:point2] = p1.genotype[point1:point2]
            p2_filtered = [gene for gene in p2.genotype if gene not in child]
            idx = 0
            for i in range(n):
                if child[i] == -1:
                    child[i] = p2_filtered[idx]
                    idx += 1
            return child

        child1_genotype = create_child(parent1, parent2)
        child2_genotype = create_child(parent2, parent1)

        return Individual(child1_genotype), Individual(child2_genotype)

    def swap_mutation(self, individual: Individual) -> Individual:
        genotype = individual.genotype.copy()
        i, j = random.sample(range(len(genotype)), 2)
        genotype[i], genotype[j] = genotype[j], genotype[i]
        return Individual(genotype)

    def inversion_mutation(self, individual: Individual) -> Individual:
        genotype = individual.genotype.copy()
        i, j = sorted(random.sample(range(len(genotype)), 2))
        genotype[i:j + 1] = reversed(genotype[i:j + 1])
        return Individual(genotype)

    def crossover_population(self, population: List[Individual]) -> List[Individual]:
        # Krzyżowanie sekwencyjne - overhead równoległości byłby za duży
        offspring = []
        n_pairs = self.population_size // 2

        for _ in range(n_pairs):
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)

            while parent1.genotype == parent2.genotype:
                parent2 = self.tournament_selection(population)

            if random.random() < self.crossover_prob:
                child1, child2 = self.order_crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])

        return offspring[:self.population_size]

    def mutate_population(self, population: List[Individual]) -> List[Individual]:
        # Mutacja sekwencyjna - również za mały overhead
        mutated = []

        for individual in population:
            if random.random() < self.mutation_prob:
                if random.random() < 0.5:
                    mutated.append(self.swap_mutation(individual))
                else:
                    mutated.append(self.inversion_mutation(individual))
            else:
                mutated.append(individual)

        return mutated

    def run(self, max_generations: int = 1000, time_limit: float = None,
            verbose: bool = True) -> Tuple[List[int], float]:
        start_time = time.time()

        self.executor = ProcessPoolExecutor(max_workers=self.n_workers)

        try:
            population = self.initialize_population()
            population = self.evaluate_population_parallel(population)

            best_individual = min(population, key=lambda ind: ind.fitness)
            self.best_solution = best_individual.genotype
            self.best_fitness = best_individual.fitness

            for generation in range(max_generations):
                if time_limit and (time.time() - start_time) > time_limit:
                    break

                offspring = self.crossover_population(population)
                offspring = self.mutate_population(offspring)
                offspring = self.evaluate_population_parallel(offspring)  # Tylko ewaluacja równolegle

                population = offspring
                current_best = min(population, key=lambda ind: ind.fitness)

                if current_best.fitness < self.best_fitness:
                    self.best_fitness = current_best.fitness
                    self.best_solution = current_best.genotype
                    if verbose:
                        print(f"Generacja {generation}: Nowa najlepsza fitness = {self.best_fitness:.2f}")

                worst_idx = max(range(len(population)), key=lambda i: population[i].fitness)
                population[worst_idx] = Individual(self.best_solution, self.best_fitness)

        finally:
            # Zamknij pool na końcu
            self.executor.shutdown()

        elapsed_time = time.time() - start_time
        if verbose:
            print(f"\nAlgorytm zakończony po {elapsed_time:.2f}s")
            print(f"Najlepsza fitness: {self.best_fitness:.2f}")

        return self.best_solution, self.best_fitness


# ============================================================================
# FUNKCJE POMOCNICZE
# ============================================================================

def generate_random_cities(n: int, seed: int = 42) -> List[City]:
    """Generuje losowe miasta"""
    random.seed(seed)
    np.random.seed(seed)
    return [City(np.random.uniform(0, 100), np.random.uniform(0, 100)) for _ in range(n)]


# ============================================================================
# GŁÓWNY PROGRAM - PORÓWNANIE
# ============================================================================

if __name__ == "__main__":
    # Parametry - zwiększone dla lepszego testu
    n_cities = 100  # Zwiększone z 50
    cities = generate_random_cities(n_cities)
    problem = TSPProblem(cities)

    print("=" * 70)
    print("PORÓWNANIE: SEKWENCYJNY vs RÓWNOLEGŁY")
    print("=" * 70)
    print(f"Problem TSP z {n_cities} miastami")
    print(f"Populacja: 200 osobników")
    print(f"Generacje: 500")
    print("=" * 70)

    # ========== WERSJA SEKWENCYJNA ==========
    print("\n>>> WERSJA SEKWENCYJNA <<<")
    print("-" * 70)

    ga_seq = SequentialGeneticAlgorithm(
        problem=problem,
        population_size=200,  # Zwiększone
        crossover_prob=0.8,
        mutation_prob=0.2,
        tournament_size=5
    )

    start_seq = time.time()
    best_route_seq, best_distance_seq = ga_seq.run(max_generations=500, verbose=False)
    time_seq = time.time() - start_seq

    print(f"Czas wykonania: {time_seq:.2f}s")
    print(f"Najlepsza fitness: {best_distance_seq:.2f}")
    print(f"Trasa: {best_route_seq[:10]}... (pierwsze 10 miast)")

    # ========== WERSJA RÓWNOLEGŁA ==========
    print("\n>>> WERSJA RÓWNOLEGŁA (4 procesy) <<<")
    print("-" * 70)

    ga_par = ParallelGeneticAlgorithm(
        problem=problem,
        population_size=200,  # Zwiększone
        crossover_prob=0.8,
        mutation_prob=0.2,
        tournament_size=5,
        n_workers=4
    )

    start_par = time.time()
    best_route_par, best_distance_par = ga_par.run(max_generations=500, verbose=False)
    time_par = time.time() - start_par

    print(f"Czas wykonania: {time_par:.2f}s")
    print(f"Najlepsza fitness: {best_distance_par:.2f}")
    print(f"Trasa: {best_route_par[:10]}... (pierwsze 10 miast)")

    # ========== PODSUMOWANIE ==========
    print("\n" + "=" * 70)
    print("PODSUMOWANIE")
    print("=" * 70)

    speedup = time_seq / time_par
    efficiency = (speedup / 4) * 100

    print(f"\nCzas:")
    print(f"  Sekwencyjny:    {time_seq:.2f}s")
    print(f"  Równoległy:     {time_par:.2f}s")
    print(f"  Przyspieszenie: {speedup:.2f}x")
    print(f"  Wydajność:      {efficiency:.1f}%")

    print(f"\nJakość rozwiązania:")
    print(f"  Sekwencyjna:  {best_distance_seq:.2f}")
    print(f"  Równoległa:   {best_distance_par:.2f}")
    difference = abs(best_distance_seq - best_distance_par)
    print(f"  Różnica:      {difference:.2f}")

    print("\n" + "=" * 70)