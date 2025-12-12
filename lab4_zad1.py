import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
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
        total += self.distance_matrix[route[-1]][route[0]]  # powrót do startu
        return total


class Individual:
    def __init__(self, genotype: List[int], fitness: float = None):
        self.genotype = genotype
        self.fitness = fitness

    def copy(self):
        return Individual(self.genotype.copy(), self.fitness)


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

    def initialize_population(self) -> List[Individual]:
        population = []

        # Część populacji losowa
        for _ in range(self.population_size // 2):
            genotype = list(range(self.problem.n))
            random.shuffle(genotype)
            population.append(Individual(genotype))

        # Część populacji z heurystyką najbliższego sąsiada
        for _ in range(self.population_size - len(population)):
            genotype = self._nearest_neighbor_heuristic()
            # Dodaj losowe zaburzenie
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

    def evaluate_population_parallel(self, population: List[Individual]) -> List[Individual]:
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            evaluated = list(executor.map(self.evaluate_individual, population))
        return evaluated

    def tournament_selection(self, population: List[Individual]) -> Individual:
        tournament = random.sample(population, self.tournament_size)
        return min(tournament, key=lambda ind: ind.fitness)

    def order_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        n = len(parent1.genotype)

        # Losowe punkty przecięcia
        point1, point2 = sorted(random.sample(range(n), 2))

        def create_child(p1, p2):
            child = [-1] * n
            # Kopiuj fragment od parent1
            child[point1:point2] = p1.genotype[point1:point2]

            # Wypełnij resztę w kolejności z parent2
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

    def crossover_population_parallel(self, population: List[Individual]) -> List[Individual]:

        def crossover_pair(pair_idx):
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)

            # Upewnij się, że rodzice są różni
            while parent1.genotype == parent2.genotype:
                parent2 = self.tournament_selection(population)

            if random.random() < self.crossover_prob:
                child1, child2 = self.order_crossover(parent1, parent2)
                return [child1, child2]
            else:
                return [parent1.copy(), parent2.copy()]

        n_pairs = self.population_size // 2

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            pairs = list(executor.map(crossover_pair, range(n_pairs)))

        offspring = []
        for pair in pairs:
            offspring.extend(pair)

        return offspring[:self.population_size]

    def mutate_population_parallel(self, population: List[Individual]) -> List[Individual]:

        def mutate_individual(individual):
            if random.random() < self.mutation_prob:
                if random.random() < 0.5:
                    return self.swap_mutation(individual)
                else:
                    return self.inversion_mutation(individual)
            return individual

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            mutated = list(executor.map(mutate_individual, population))

        return mutated

    def run(self, max_generations: int = 1000, time_limit: float = None,
            verbose: bool = True) -> Tuple[List[int], float]:
        start_time = time.time()

        # Inicjalizacja
        population = self.initialize_population()
        population = self.evaluate_population_parallel(population)

        # Znajdź najlepszego
        best_individual = min(population, key=lambda ind: ind.fitness)
        self.best_solution = best_individual.genotype
        self.best_fitness = best_individual.fitness

        for generation in range(max_generations):
            # Sprawdź limit czasu
            if time_limit and (time.time() - start_time) > time_limit:
                break

            # RÓWNOLEGŁA selekcja i krzyżowanie
            offspring = self.crossover_population_parallel(population)

            # RÓWNOLEGŁA mutacja
            offspring = self.mutate_population_parallel(offspring)

            # RÓWNOLEGŁA ewaluacja potomków
            offspring = self.evaluate_population_parallel(offspring)

            # Elityzm - zachowaj najlepszego
            population = offspring
            current_best = min(population, key=lambda ind: ind.fitness)

            if current_best.fitness < self.best_fitness:
                self.best_fitness = current_best.fitness
                self.best_solution = current_best.genotype
                if verbose:
                    print(f"Generacja {generation}: Nowa najlepsza fitness = {self.best_fitness:.2f}")

            # Dodaj najlepszego z poprzedniej generacji (elityzm)
            worst_idx = max(range(len(population)), key=lambda i: population[i].fitness)
            population[worst_idx] = Individual(self.best_solution, self.best_fitness)

        elapsed_time = time.time() - start_time
        if verbose:
            print(f"\nAlgorytm zakończony po {elapsed_time:.2f}s")
            print(f"Najlepsza fitness: {self.best_fitness:.2f}")

        return self.best_solution, self.best_fitness


# Funkcja generująca przykładowe dane
def generate_random_cities(n: int, seed: int = 42) -> List[City]:
    random.seed(seed)
    np.random.seed(seed)
    return [City(np.random.uniform(0, 100), np.random.uniform(0, 100)) for _ in range(n)]


# Przykład użycia
if __name__ == "__main__":
    # Generuj problem
    n_cities = 50
    cities = generate_random_cities(n_cities)
    problem = TSPProblem(cities)

    print(f"Problem TSP z {n_cities} miastami")
    print("=" * 50)

    # Uruchom algorytm równoległy
    ga = ParallelGeneticAlgorithm(
        problem=problem,
        population_size=100,
        crossover_prob=0.8,
        mutation_prob=0.2,
        tournament_size=5,
        n_workers=4
    )

    best_route, best_distance = ga.run(max_generations=500, verbose=True)

    print(f"\nNajlepsza trasa: {best_route[:10]}... (pierwsze 10 miast)")
    print(f"Długość trasy: {best_distance:.2f}")