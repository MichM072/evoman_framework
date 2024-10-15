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
from logger import Logger


class GASpecialistSA:
    MODE_TRAIN = "Train"
    MODE_TUNE = "Tune"
    MODE_TEST = "Test"

    def __init__(
        self,
        sa: bool = False,
        headless: bool = True,
        experiment_name: str = "optimization_specialist_sa_group55",
        n_hidden_neurons: int = 10,
        population_size: int = 100,
        generations: int = 50,
        crossover_probability: float = 0.5,
        init_t: float = 100,
        min_t: float = 1,
        max_mutpb: float = 0.5,
        min_mutpb: float = 0.01,
        cooling_rate: float = 0.99,
    ):
        self.headless = headless
        self.experiment_name = experiment_name
        self.n_hidden_neurons = n_hidden_neurons
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability

        # Simulated Annealing Hyperparameters
        self.init_t = init_t  # Starting temp
        self.min_t = min_t  # Minimum Temp
        self.max_mutpb = max_mutpb  # Max mutation probability
        self.min_mutpb = min_mutpb  # Min mutation probability
        self.cooling_rate = cooling_rate  # Cooling Rate

        self.enemy = 4  # default placeholder
        self.env = None
        self.run = 0
        self.mode = self.MODE_TRAIN
        self.sa = sa  # Simulated Annealing enabled or disabled

        # other
        self.logger = None

    # Environment Setup
    def setup_environment(self, enemy: int) -> Environment:
        # Headless meaning the experiment runs faster
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        # Creates folder in path if does not exist
        if not os.path.exists(self.experiment_name):
            os.makedirs(self.experiment_name)

        return Environment(
            experiment_name=self.experiment_name,
            enemies=[enemy],
            playermode="ai",
            player_controller=player_controller(self.n_hidden_neurons),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False,
            randomini="yes",
        )

    def setup_deap(
            self,
            env: Environment,
            n_vars: int
    ) -> Toolbox:
        # Sets up DEAP package and GA
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
        random_numbers = []

        for _ in range(n_vars):
            random_number = random.uniform(-1, 1)
            random_numbers.append(random_number)

        return random_numbers

    def simulation(
            self,
            env: Environment,
            individual: np.array
    ) -> tuple[any]:
        fitness, _, _, _ = env.play(pcont=np.array(individual))
        return (fitness,)

    def run_evolution(self, toolbox: base) -> None:
        # Runs the evolution process using the genetic algorithm.
        population = toolbox.population(n=self.population_size)
        best = tools.HallOfFame(1) # the best individual that ever lived in the population during the evolution

        # Set current Temp
        T = self.init_t

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        self.logger.log_initial_population_fitness()

        # Assign a function to the probability if we use SA else use default mutation probability
        mutpb_func = lambda: (
            self.calculate_mutation_probability_SA(T) if self.sa else self.max_mutpb
        )

        for generation in range(self.generations):
            self.evaluate_population(population, toolbox)

            # Calculate mutation probability
            mutation_prob = mutpb_func()

            print(f"Mutation prob: {mutation_prob}")
            print(f"Temperature: {T}")

            # Record statistics
            record = stats.compile(population)
            best.update(population)

            self.logger.log_generation_statistics(generation, best, record)

            # Generate the next generation
            offspring = self.generate_offspring(population, toolbox, mutation_prob)

            # Replace the population with the offspring
            population[:] = offspring

            # Apply cooling scheme
            # Variant used: Exponential
            T = T * self.cooling_rate

        self.logger.save_results(best, self.run)
        self.env.state_to_log()

    def calculate_mutation_probability_SA(self, T: float):
        return self.min_mutpb + (self.max_mutpb - self.min_mutpb) * (T / self.init_t)

    def evaluate_population(
        self,
        population: np.array,
        toolbox: base
    ) -> None:
        # Evaluates the fitness of the entire population
        fitness = list(map(toolbox.evaluate, population))

        for ind, fit in zip(population, fitness):
            ind.fitness.values = fit

    def generate_offspring(
        self,
        population: np.array,
        toolbox: base,
        mutation_prob: float
    ) -> list:
        # Select individuals from pop
        offspring = toolbox.select(population, len(population))
        # Make copies of the pop
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Apply crossover to pairs of inds based on crossover_probability
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < self.crossover_probability:
                toolbox.mate(offspring[i], offspring[i + 1])

                # After crossover, the fitness is deleted as the child "changed".
                # So the fitness is not correct anymore
                # The children are evaluated
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Apply mutation to individuals with some probability
        for ind in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(ind)

                del ind.fitness.values

        # Recalculate fitness for individuals whose fitness was deleted
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness_values = map(toolbox.evaluate, invalid_ind)

        # Assign the recalculated fitness values to the individuals
        for ind, fit in zip(invalid_ind, fitness_values):
            ind.fitness.values = fit

        return offspring

    def run_experiment(
        self,
        enemy: int,
        mode: str,
        run: int,
        best_ind_idx: int
    ):
        self.enemy = enemy
        self.run = run
        self.mode = mode

        start_time = time.time()

        self.env = self.setup_environment(enemy=self.enemy)
        n_vars = (self.env.get_num_sensors() + 1) * self.n_hidden_neurons + (self.n_hidden_neurons + 1) * 5

        self.logger = Logger(
            self.experiment_name,
            self.mode,
            self.enemy
        )

        """
           - TRAIN or TUNE: Runs the evolution process
           - TEST: Runs the simulation using the best solution from memory
        """
        if self.mode == self.MODE_TRAIN or self.MODE_TUNE:
            toolbox = self.setup_deap(self.env, n_vars)
            self.run_evolution(toolbox)
        elif self.mode == self.MODE_TEST:
            best_ind = np.loadtxt(self.experiment_name+f'/enemy_{self.enemy}/best_{best_ind_idx}.txt')
            print("\n Using best solution from memory \n")
            fitness, player, enemy, game_time = self.env.play(pcont=best_ind)
            return fitness, player, enemy, game_time
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        end_time = time.time()

        print(f"\nExecution time: {round((end_time - start_time) / 60)} minutes")
