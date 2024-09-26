###############################################################################
# EvoMan FrameWork - V1.0 2016                                                #
# DEMO: Basic Genetic Algorithm (with DEAP).                                  #
# Author: Group 55                                                            #
###############################################################################

# Imports
import os
import random
import time
import numpy as np
from deap import base, creator, tools
from deap.base import Toolbox

from evoman.environment import Environment
from demo_controller import player_controller

HEADLESS = True
EXPERIMENT_NAME = "optimization_specialist_SA_group55"
N_HIDDEN_NEURONS = 10
POPULATION_SIZE = 100
GENERATIONS = 50
CROSSOVER_PROBABILITY = 0.5

# Simulated Annealing Hyper parameters
INIT_T = 100  # Starting temp
MIN_T = 1  # Minimum Temp
MAX_MUTPB = 0.5  # Max mutation probability
MIN_MUTPB = 0.01  # Min mutation probability
COOLING_RATE = 0.5  # Cooling Rate


class GASpecialistSA:
    def __init__(self, SA: bool = False,
                 experiment_name: str = EXPERIMENT_NAME):
        self.enemy = 4  # default placeholder
        self.env = ""
        self.SA = SA  # Simulated Annealing enabled or disabled
        self.mode = "Train"
        self.experiment_name = experiment_name

    # Environment Setup
    def setup_environment(self, enemy: int) -> Environment:
        # Headless meaning the experiment runs faster
        if HEADLESS:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        # Creates folder in path if does not exist
        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

        return Environment(
            experiment_name=self.experiment_name,
            enemies=[enemy],
            playermode="ai",
            player_controller=player_controller(N_HIDDEN_NEURONS),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False,
            randomini="yes",
        )

    def setup_deap(self, env: Environment) -> Toolbox:
        # Sets up DEAP's genetic algorithm
        n_vars = (env.get_num_sensors() + 1) * N_HIDDEN_NEURONS + (
            N_HIDDEN_NEURONS + 1
        ) * 5

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,
            lambda: self.generate_individual(n_vars),
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.simulation, env)
        toolbox.register(
            "mate",
            tools.cxTwoPoint,
        )
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        return toolbox

    def generate_individual(self, n_vars: int) -> list[float]:
        # Generates an individual with random weights within the domain limits.
        return [random.uniform(-1, 1) for _ in range(n_vars)]

    def simulation(self, env: Environment, individual: np.array) -> tuple[any]:
        # Evaluates an individual's fitness in the environment.
        fitness, _, _, _ = env.play(pcont=np.array(individual))
        return (fitness,)

    def apply_limits(self, individual: np.array) -> np.array:
        # Applies limits to the individual's gene values.
        return [max(min(gene, 1), -1) for gene in individual]

    def run_evolution(self, toolbox) -> None:
        # Runs the evolution process using the genetic algorithm.
        population = toolbox.population(n=POPULATION_SIZE)
        best = tools.HallOfFame(1)

        # Set current Temp
        T = INIT_T

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        self.log_initial_population_fitness()

        # Assign a function to the probability if we use SA else use default mutation proability
        mutpb_func = lambda: (
            self.calculate_mutation_probability_SA(T) if self.SA else MAX_MUTPB
        )

        for generation in range(GENERATIONS):
            self.evaluate_population(population, toolbox)

            # Calculate mutation probability
            mutation_prob = mutpb_func()

            print(f"Mutation prob: {mutation_prob}")
            print(f"Temperature: {T}")

            # Record statistics
            record = stats.compile(population)
            best.update(population)

            self.log_generation_statistics(generation, best, record)

            # Generate the next generation
            offspring = self.generate_offspring(population, toolbox, mutation_prob)

            # Replace the population with the offspring
            population[:] = offspring

            # Apply cooling scheme
            # Variant used: Exponential
            T = T * COOLING_RATE

        self.save_results(best)

    def calculate_mutation_probability_SA(self, T):
        return MIN_MUTPB + (MAX_MUTPB - MIN_MUTPB) * (T / INIT_T)

    def log_initial_population_fitness(self) -> None:
        # Logs the initial population's fitness to a file.
        with open(self.experiment_name + "/results.txt", "a") as file_aux:
            file_aux.write(
                "\n{:<10} {:<10} {:<10} {:<10} ENEMY: {}\n".format(
                    "GENERATION", "BEST", "MEAN", "STD", self.enemy
                )
            )

    def evaluate_population(self, population: np.array, toolbox) -> None:
        # Evaluates the fitness of the entire population.
        fitness = list(map(toolbox.evaluate, population))

        for ind, fit in zip(population, fitness):
            ind.fitness.values = fit

    def log_generation_statistics(
        self, generation: np.array, best: np.array, record: np.array
    ) -> None:
        # Logs the statistics for the current generation.
        best_ind = best[0]
        best_fitness = best_ind.fitness.values[0]

        print(
            f'\n GENERATION {generation} best: {round(best_fitness, 6)} avg: {round(record["avg"], 6)} std: {round(record["std"], 6)} enemy: {self.enemy}'
        )

        with open(self.experiment_name + "/results.txt", "a") as file_aux:
            file_aux.write(
                f'\n{generation:<10} {best_fitness:<10.6f} {record["avg"]:<10.6f} {record["std"]:<10.6f}'
            )

    def generate_offspring(
        self, population: np.array, toolbox, mutation_prob: float
    ) -> list:
        # Generates offspring through selection, crossover, and mutation.
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_PROBABILITY:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for child in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(child)
                del child.fitness.values

        # Evaluate individuals with invalid fitness values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness = map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        return offspring

    def save_results(self, best) -> None:
        # Saves the best solution and logs the simulation state.
        np.savetxt(self.experiment_name + f"/best_{self.enemy}.txt", best[0])
        print(f"\nBest fitness achieved: {best[0].fitness.values[0]}")

        # Save the simulation state and log state of the environment
        self.env.state_to_log()

    def run_experiment(self, enemy: int, mode: str):
        self.enemy = enemy
        self.mode = mode
        ini = time.time()
        self.env = self.setup_environment(enemy=self.enemy)

        if self.mode == "Train":
            toolbox = self.setup_deap(self.env)
            self.run_evolution(toolbox)
        elif self.mode == "Test":
            # Run simulation with the best solution for selected enemy.
            best_ind = np.loadtxt(self.experiment_name+f'/best_{self.enemy}.txt')
            print("\n Using best solution from memory \n")
            fitness,player,enemy,game_time = self.env.play(pcont=best_ind)
            return fitness,player,enemy,game_time
        else:
            print(f"Invalid mode: {self.mode}")

        end_time = time.time()

        print(f"\nExecution time: {round((end_time - ini) / 60)} minutes")
