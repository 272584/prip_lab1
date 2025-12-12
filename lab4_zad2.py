import numpy as np
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import random
from multiprocessing import Manager, Queue
import threading


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

    def evaluate_route_vectorized(self, route: np.ndarray) -> float:
        """Wektoryzowana wersja ewaluacji - akceleracja GPU-like"""
        indices = np.append(route, route[0])
        distances = self.distance_matrix[indices[:-1], indices[1:]]
        return np.sum(distances)


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


def evaluate_batch_vectorized(args):
    """Wektoryzowana ewaluacja - wykorzystanie NumPy dla akceleracji"""
    individuals, distance_matrix = args
    results = []

    # Konwersja do NumPy array dla szybszych obliczeń
    distance_matrix_np = np.array(distance_matrix)

    for individual in individuals:
        if individual.fitness is None:
            route = np.array(individual.genotype)
            # Wektoryzowana ewaluacja
            indices = np.append(route, route[0])
            distances = distance_matrix_np[indices[:-1], indices[1:]]
            individual.fitness = np.sum(distances)
        results.append(individual)
    return results


def run_island_ga(args):
    """Funkcja uruchamiająca GA na pojedynczej wyspie"""
    island_id, problem, population_size, generations, migration_queue = args

    # Inicjalizacja GA dla wyspy
    ga = IslandGeneticAlgorithm(
        problem=problem,
        population_size=population_size,
        island_id=island_id,
        migration_queue=migration_queue
    )

    return ga.run(max_generations=generations, verbose=False)


# ============================================================================
# WERSJA SEKWENCYJNA (baseline)
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
        for individual in population:
            self.evaluate_individual(individual)
        return population

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
# WERSJA Z PROCESAMI (z zadania 1)
# ============================================================================

class ParallelGeneticAlgorithm(SequentialGeneticAlgorithm):
    def __init__(self, problem: TSPProblem,
                 population_size: int = 100,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.2,
                 tournament_size: int = 3,
                 n_workers: int = 4):
        super().__init__(problem, population_size, crossover_prob, mutation_prob, tournament_size)
        self.n_workers = n_workers
        self.executor = None

    def evaluate_population_parallel(self, population: List[Individual]) -> List[Individual]:
        batch_size = len(population) // self.n_workers
        if batch_size == 0:
            batch_size = 1

        batches = []
        for i in range(0, len(population), batch_size):
            batch = population[i:i + batch_size]
            batches.append((batch, self.problem.distance_matrix))

        results = list(self.executor.map(evaluate_batch, batches))

        evaluated = []
        for batch_result in results:
            evaluated.extend(batch_result)

        return evaluated

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
                offspring = self.evaluate_population_parallel(offspring)

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
            self.executor.shutdown()

        elapsed_time = time.time() - start_time
        if verbose:
            print(f"\nAlgorytm zakończony po {elapsed_time:.2f}s")
            print(f"Najlepsza fitness: {self.best_fitness:.2f}")

        return self.best_solution, self.best_fitness


# ============================================================================
# ZADANIE 2a: WERSJA Z AKCELERACJĄ GPU (NumPy vectorization)
# ============================================================================

class GPUAcceleratedGeneticAlgorithm(ParallelGeneticAlgorithm):

    def evaluate_population_gpu(self, population: List[Individual]) -> List[Individual]:
        batch_size = len(population) // self.n_workers
        if batch_size == 0:
            batch_size = 1

        batches = []
        for i in range(0, len(population), batch_size):
            batch = population[i:i + batch_size]
            batches.append((batch, self.problem.distance_matrix))

        # Używamy wektoryzowanej wersji
        results = list(self.executor.map(evaluate_batch_vectorized, batches))

        evaluated = []
        for batch_result in results:
            evaluated.extend(batch_result)

        return evaluated

    def run(self, max_generations: int = 1000, time_limit: float = None,
            verbose: bool = True) -> Tuple[List[int], float]:
        start_time = time.time()

        self.executor = ProcessPoolExecutor(max_workers=self.n_workers)

        try:
            population = self.initialize_population()
            population = self.evaluate_population_gpu(population)  # GPU-accelerated

            best_individual = min(population, key=lambda ind: ind.fitness)
            self.best_solution = best_individual.genotype
            self.best_fitness = best_individual.fitness

            for generation in range(max_generations):
                if time_limit and (time.time() - start_time) > time_limit:
                    break

                offspring = self.crossover_population(population)
                offspring = self.mutate_population(offspring)
                offspring = self.evaluate_population_gpu(offspring)  # GPU-accelerated

                population = offspring
                current_best = min(population, key=lambda ind: ind.fitness)

                if current_best.fitness < self.best_fitness:
                    self.best_fitness = current_best.fitness
                    self.best_solution = current_best.genotype
                    if verbose:
                        print(f"[GPU] Generacja {generation}: Nowa najlepsza fitness = {self.best_fitness:.2f}")

                worst_idx = max(range(len(population)), key=lambda i: population[i].fitness)
                population[worst_idx] = Individual(self.best_solution, self.best_fitness)

        finally:
            self.executor.shutdown()

        elapsed_time = time.time() - start_time
        if verbose:
            print(f"\n[GPU] Algorytm zakończony po {elapsed_time:.2f}s")
            print(f"[GPU] Najlepsza fitness: {self.best_fitness:.2f}")

        return self.best_solution, self.best_fitness


# ============================================================================
# ZADANIE 2b: MODEL WYSPOWY (Distributed Island Model)
# ============================================================================

class IslandGeneticAlgorithm(SequentialGeneticAlgorithm):

    def __init__(self, problem: TSPProblem,
                 population_size: int = 50,
                 island_id: int = 0,
                 migration_queue: Optional[Queue] = None,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.2,
                 tournament_size: int = 3):
        super().__init__(problem, population_size, crossover_prob, mutation_prob, tournament_size)
        self.island_id = island_id
        self.migration_queue = migration_queue
        self.immigrants = []

    def migrate_individuals(self, population: List[Individual], n_migrants: int = 2):
        if self.migration_queue is None:
            return

        # Sortuj populację i wybierz najlepszych
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
        migrants = sorted_pop[:n_migrants]

        # Wyślij migrantów
        for migrant in migrants:
            try:
                self.migration_queue.put((self.island_id, migrant.copy()), block=False)
            except:
                pass  # Kolejka pełna, ignoruj

    def receive_immigrants(self, population: List[Individual]):
        """Odbiera imigrantów z innych wysp"""
        if self.migration_queue is None:
            return population

        # Odbierz wszystkich dostępnych imigrantów
        immigrants = []
        while True:
            try:
                source_island, immigrant = self.migration_queue.get(block=False)
                if source_island != self.island_id:  # Nie przyjmuj własnych migrantów
                    immigrants.append(immigrant)
            except:
                break

        if not immigrants:
            return population

        # Zastąp najgorszych osobników imigrantami
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        n_replace = min(len(immigrants), len(population) // 4)  # Zastąp max 25%

        for i in range(n_replace):
            sorted_pop[i] = immigrants[i]

        return sorted_pop

    def run(self, max_generations: int = 1000, migration_interval: int = 50,
            time_limit: float = None, verbose: bool = True) -> Tuple[List[int], float]:
        start_time = time.time()

        population = self.initialize_population()
        population = self.evaluate_population(population)

        best_individual = min(population, key=lambda ind: ind.fitness)
        self.best_solution = best_individual.genotype
        self.best_fitness = best_individual.fitness

        for generation in range(max_generations):
            if time_limit and (time.time() - start_time) > time_limit:
                break

            # Migracja co migration_interval pokoleń
            if generation > 0 and generation % migration_interval == 0:
                self.migrate_individuals(population)
                population = self.receive_immigrants(population)

            offspring = self.crossover_population(population)
            offspring = self.mutate_population(offspring)
            offspring = self.evaluate_population(offspring)

            population = offspring
            current_best = min(population, key=lambda ind: ind.fitness)

            if current_best.fitness < self.best_fitness:
                self.best_fitness = current_best.fitness
                self.best_solution = current_best.genotype
                if verbose:
                    print(f"[Wyspa {self.island_id}] Gen {generation}: fitness = {self.best_fitness:.2f}")

            worst_idx = max(range(len(population)), key=lambda i: population[i].fitness)
            population[worst_idx] = Individual(self.best_solution, self.best_fitness)

        elapsed_time = time.time() - start_time
        if verbose:
            print(f"\n[Wyspa {self.island_id}] Zakończono po {elapsed_time:.2f}s")
            print(f"[Wyspa {self.island_id}] Najlepsza fitness: {self.best_fitness:.2f}")

        return self.best_solution, self.best_fitness


class DistributedIslandGA:
    """
    Koordynator modelu wyspowego - zarządza wieloma wyspami
    """

    def __init__(self, problem: TSPProblem,
                 n_islands: int = 4,
                 population_per_island: int = 50,
                 n_workers: int = 4):
        self.problem = problem
        self.n_islands = n_islands
        self.population_per_island = population_per_island
        self.n_workers = n_workers

    def run(self, max_generations: int = 1000, migration_interval: int = 50,
            verbose: bool = True) -> Tuple[List[int], float]:
        start_time = time.time()

        # Utwórz kolejkę do migracji (shared memory)
        manager = Manager()
        migration_queue = manager.Queue(maxsize=100)

        # Przygotuj argumenty dla każdej wyspy
        island_args = []
        for island_id in range(self.n_islands):
            island_args.append((
                island_id,
                self.problem,
                self.population_per_island,
                max_generations,
                migration_queue
            ))

        # Uruchom wyspy równolegle
        print(f"\n{'=' * 70}")
        print(f"MODEL WYSPOWY: {self.n_islands} wyspy, {self.population_per_island} osobników/wyspa")
        print(f"{'=' * 70}\n")

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Każda wyspa działa niezależnie
            results = list(executor.map(run_island_ga, island_args))

        # Znajdź najlepsze rozwiązanie ze wszystkich wysp
        best_solution = None
        best_fitness = float('inf')

        for island_id, (solution, fitness) in enumerate(results):
            if verbose:
                print(f"Wyspa {island_id}: fitness = {fitness:.2f}")
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = solution

        elapsed_time = time.time() - start_time

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"NAJLEPSZE ROZWIĄZANIE ze wszystkich wysp")
            print(f"Fitness: {best_fitness:.2f}")
            print(f"Czas: {elapsed_time:.2f}s")
            print(f"{'=' * 70}\n")

        return best_solution, best_fitness


# ============================================================================
# FUNKCJE POMOCNICZE
# ============================================================================

def generate_random_cities(n: int, seed: int = 42) -> List[City]:
    """Generuje losowe miasta"""
    random.seed(seed)
    np.random.seed(seed)
    return [City(np.random.uniform(0, 100), np.random.uniform(0, 100)) for _ in range(n)]


# ============================================================================
# GŁÓWNY PROGRAM - PORÓWNANIE WSZYSTKICH WERSJI
# ============================================================================

if __name__ == "__main__":
    # Parametry
    n_cities = 100
    cities = generate_random_cities(n_cities)
    problem = TSPProblem(cities)

    print("=" * 70)
    print("PORÓWNANIE WSZYSTKICH IMPLEMENTACJI")
    print("=" * 70)
    print(f"Problem TSP z {n_cities} miastami")
    print(f"Populacja: 200 osobników")
    print(f"Generacje: 300")
    print("=" * 70)

    results = {}

    # ========== 1. WERSJA SEKWENCYJNA ==========
    print("\n>>> 1. WERSJA SEKWENCYJNA (baseline) <<<")
    print("-" * 70)

    ga_seq = SequentialGeneticAlgorithm(
        problem=problem,
        population_size=200,
        crossover_prob=0.8,
        mutation_prob=0.2,
        tournament_size=5
    )

    start = time.time()
    best_route, best_fitness = ga_seq.run(max_generations=300, verbose=False)
    time_seq = time.time() - start

    results['Sequential'] = (time_seq, best_fitness)
    print(f"Czas: {time_seq:.2f}s, Fitness: {best_fitness:.2f}")

    # ========== 2. WERSJA RÓWNOLEGŁA (ProcessPool) ==========
    print("\n>>> 2. WERSJA RÓWNOLEGŁA (4 procesy) <<<")
    print("-" * 70)

    ga_par = ParallelGeneticAlgorithm(
        problem=problem,
        population_size=200,
        crossover_prob=0.8,
        mutation_prob=0.2,
        tournament_size=5,
        n_workers=4
    )

    start = time.time()
    best_route, best_fitness = ga_par.run(max_generations=300, verbose=False)
    time_par = time.time() - start

    results['Parallel'] = (time_par, best_fitness)
    print(f"Czas: {time_par:.2f}s, Fitness: {best_fitness:.2f}")

    # ========== 3. WERSJA GPU-ACCELERATED ==========
    print("\n>>> 3. WERSJA Z AKCELERACJĄ GPU (NumPy vectorization) <<<")
    print("-" * 70)

    ga_gpu = GPUAcceleratedGeneticAlgorithm(
        problem=problem,
        population_size=200,
        crossover_prob=0.8,
        mutation_prob=0.2,
        tournament_size=5,
        n_workers=4
    )

    start = time.time()
    best_route, best_fitness = ga_gpu.run(max_generations=300, verbose=False)
    time_gpu = time.time() - start

    results['GPU-Accelerated'] = (time_gpu, best_fitness)
    print(f"Czas: {time_gpu:.2f}s, Fitness: {best_fitness:.2f}")

    # ========== 4. MODEL WYSPOWY ==========
    print("\n>>> 4. MODEL WYSPOWY (Distributed) <<<")
    print("-" * 70)

    ga_island = DistributedIslandGA(
        problem=problem,
        n_islands=4,
        population_per_island=50,  # 4 * 50 = 200 łącznie
        n_workers=4
    )

    start = time.time()
    best_route, best_fitness = ga_island.run(
        max_generations=300,
        migration_interval=50,
        verbose=False
    )
    time_island = time.time() - start

    results['Island Model'] = (time_island, best_fitness)
    print(f"Czas: {time_island:.2f}s, Fitness: {best_fitness:.2f}")

    # ========== PODSUMOWANIE ==========
    print("\n" + "=" * 70)
    print("PODSUMOWANIE WYNIKÓW")
    print("=" * 70)

    print(f"\n{'Metoda':<25} {'Czas [s]':<12} {'Fitness':<12} {'Speedup':<10}")
    print("-" * 70)

    for method, (t, f) in results.items():
        speedup = time_seq / t
        print(f"{method:<25} {t:>10.2f}s  {f:>10.2f}  {speedup:>8.2f}x")
