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
EXPERIMENT_NAME = 'optimization_specialist_group55'
N_HIDDEN_NEURONS = 10
POPULATION_SIZE = 100
GENERATIONS = 30
MUTATION_PROBABILITY = 0.2
CROSSOVER_PROBABILITY = 0.5

# Environment Setup
def setup_environment(enemy: int) -> Environment:
    # Headless meaning the experiment runs faster
    if HEADLESS:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Creates folder in path if does not exist
    if not os.path.exists(EXPERIMENT_NAME):
        os.makedirs(EXPERIMENT_NAME)

    return Environment(
        experiment_name=EXPERIMENT_NAME,
        enemies=[enemy],
        playermode="ai",
        player_controller=player_controller(N_HIDDEN_NEURONS),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False
    )

def setup_deap(env: Environment) -> Toolbox:
    # Sets up DEAP's genetic algorithm
    n_vars = (env.get_num_sensors() + 1) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5

    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('individual', tools.initIterate, creator.Individual, lambda: generate_individual(n_vars))
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', simulation, env)
    toolbox.register('mate', tools.cxTwoPoint,)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.2)
    toolbox.register('select', tools.selTournament, tournsize=3)

    return toolbox

def generate_individual(n_vars: int) -> list[float]:
    # Generates an individual with random weights within the domain limits.
    return [random.uniform(-1, 1) for _ in range(n_vars)]

def simulation(env: Environment, individual: np.array) -> tuple[any]:
    # Evaluates an individual's fitness in the environment.
    fitness, _, _, _ = env.play(pcont=np.array(individual))
    return (fitness,)

def apply_limits(individual: np.array) -> np.array:
    # Applies limits to the individual's gene values.
    return [max(min(gene, 1), -1) for gene in individual]

def run_evolution(toolbox) -> None:
    # Runs the evolution process using the genetic algorithm.
    population = toolbox.population(n=POPULATION_SIZE)
    best = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)

    log_initial_population_fitness()

    for generation in range(GENERATIONS):
        evaluate_population(population, toolbox)

        # Record statistics
        record = stats.compile(population)
        best.update(population)

        log_generation_statistics(generation, best, record)

        # Generate the next generation
        offspring = generate_offspring(population, toolbox)

        # Replace the population with the offspring
        population[:] = offspring

    save_results(best)

def log_initial_population_fitness() -> None:
    # Logs the initial population's fitness to a file.
    with open(EXPERIMENT_NAME + '/results.txt', 'a') as file_aux:
        file_aux.write('\n\ngeneration best mean std\n')

def evaluate_population(population: np.array, toolbox) -> None:
    # Evaluates the fitness of the entire population.
    fitness = list(map(toolbox.evaluate, population))

    for ind, fit in zip(population, fitness):
        ind.fitness.values = fit

def log_generation_statistics(
        generation: np.array,
        best: np.array,
        record: np.array) -> None:
    # Logs the statistics for the current generation.
    best_ind = best[0]
    best_fitness = best_ind.fitness.values[0]

    print(
        f'\n GENERATION {generation} best: {round(best_fitness, 6)} avg: {round(record["avg"], 6)} std: {round(record["std"], 6)}')

    with open(EXPERIMENT_NAME + '/results.txt', 'a') as file_aux:
        file_aux.write(f'\n{generation} {round(best_fitness, 6)} {round(record["avg"], 6)} {round(record["std"], 6)}')

def generate_offspring(population: np.array, toolbox) -> list:
    # Generates offspring through selection, crossover, and mutation.
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CROSSOVER_PROBABILITY:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for child in offspring:
        if random.random() < MUTATION_PROBABILITY:
            toolbox.mutate(child)
            del child.fitness.values

    # Evaluate individuals with invalid fitness values
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitness = map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitness):
        ind.fitness.values = fit

    return offspring

def save_results(best) -> None:
    # Saves the best solution and logs the simulation state.
    np.savetxt(EXPERIMENT_NAME + '/best.txt', best[0])
    print(f'\nBest fitness achieved: {best[0].fitness.values[0]}')

    # Save the simulation state and log state of the environment
    env.state_to_log()

# Main Execution
if __name__ == "__main__":
    ini = time.time()
    env = setup_environment(enemy=4)
    toolbox = setup_deap(env)

    run_evolution(toolbox)

    end_time = time.time()

    print(f'\nExecution time: {round((end_time - ini) / 60)} minutes')