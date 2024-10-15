import os
import csv
import sys
import numpy as np
import multiprocessing as mp
from deap import base, creator, tools, algorithms
from evoman.environment import Environment
from demo_controller import player_controller

# Set headless mode to speed up simulations
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Constants and configurations
N_HIDDEN_NEURONS = 10
N_POPULATION = 100
N_GENERATIONS = 30
MUTATION_PROBABILITY = 0.1
CROSSOVER_PROBABILITY = 0.9
COOLING_RATE = 0.05
ENEMIES = [2, 4, 8]
TRAIN_RUNS = 10
TEST_RUNS = 5

MODE_TRAIN = 'Train'
MODE_TEST = 'Test'
MODE = MODE_TEST  # 'Train' or 'Test'


def create_environment(n_hidden_neurons, experiment_name, enemy):
    return Environment(
        experiment_name=experiment_name,
        enemies=[enemy],
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False,
        randomini='yes'
    )


def initialize_deap_toolbox(n_vars, env):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual, env)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def evaluate_individual(env, individual):
    return simulate_environment(env, individual),


def simulate_environment(env, individual):
    fitness, _, _, _ = env.play(pcont=individual)
    return fitness


def custom_similar(ind1, ind2):
    """Custom similarity function for Hall of Fame comparison."""
    return np.array_equal(ind1, ind2)


def run_ea_ga(toolbox, experiment_name, env):
    """Run the Genetic Algorithm (EA1) without simulated annealing."""
    population = toolbox.population(n=N_POPULATION)
    hof = tools.HallOfFame(1, similar=custom_similar)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    population, log = algorithms.eaSimple(
        population, toolbox, cxpb=CROSSOVER_PROBABILITY, mutpb=MUTATION_PROBABILITY,
        ngen=N_GENERATIONS, stats=stats, halloffame=hof, verbose=True
    )

    save_best_solution(hof[0], experiment_name, 'best.txt')
    log_results_to_csv(log, experiment_name, 'results_ga.csv')
    env.update_solutions([population, [ind.fitness.values[0] for ind in population]])
    env.save_state()

    # print(f"Best individual (GA): {hof[0]}")


def run_ea_ga_sa(toolbox, experiment_name, env):
    """Run the Genetic Algorithm with Simulated Annealing (EA2)."""
    population = toolbox.population(n=N_POPULATION)
    hof = tools.HallOfFame(1, similar=custom_similar)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    log_file_path = f"{experiment_name}/results_ga_sa.csv"

    with open(log_file_path, 'w', newline='') as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(['Generation', 'Max', 'Avg', 'Std', 'Min'])

        for gen in range(N_GENERATIONS + 1):
            dynamic_mutpb = MUTATION_PROBABILITY * np.exp(-COOLING_RATE * gen)

            # Apply genetic algorithm steps manually to include custom mutation probability
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < CROSSOVER_PROBABILITY:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.rand() < dynamic_mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate individuals with invalid fitness
            invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_individuals)
            for ind, fit in zip(invalid_individuals, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring
            hof.update(population)

            record = stats.compile(population)
            log_generation_to_csv(csv_writer, gen, record)

    save_best_solution(hof[0], experiment_name, 'best_sa.txt')
    env.update_solutions([population, [ind.fitness.values[0] for ind in population]])
    env.save_state()

    # print(f"Best individual (GA+SA): {hof[0]}")


def log_generation_to_csv(csv_writer, generation, record):
    csv_writer.writerow([generation, record['max'], record['avg'], record['std'], record['min']])
    # print(f"Generation {generation}: Max {record['max']} Avg {record['avg']} Std {record['std']} Min {record['min']}")


def save_best_solution(solution, experiment_name, file_name):
    np.savetxt(f'{experiment_name}/{file_name}', solution)


def log_results_to_csv(log, experiment_name, file_name):
    with open(f'{experiment_name}/{file_name}', 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Generation', 'Max', 'Avg', 'Std', 'Min'])
        for generation in log:
            csv_writer.writerow(
                [generation['gen'], generation['max'], generation['avg'], generation['std'], generation['min']])


def train_ga(i, enemy):
    """Train using the basic Genetic Algorithm (EA1)."""
    experiment_name = f'train_run_enemy{enemy}/GA_train_run{i + 1}_enemy{enemy}'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    env = create_environment(N_HIDDEN_NEURONS, experiment_name, enemy)
    n_vars = calculate_n_vars(env)
    toolbox = initialize_deap_toolbox(n_vars, env)
    run_ea_ga(toolbox, experiment_name, env)


def train_ga_sa(i, enemy):
    """Train using the Genetic Algorithm with Simulated Annealing (EA2)."""
    experiment_name = f'train_run_enemy{enemy}/GA_SA_train_run{i + 1}_enemy{enemy}'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    env = create_environment(N_HIDDEN_NEURONS, experiment_name, enemy)
    n_vars = calculate_n_vars(env)
    toolbox = initialize_deap_toolbox(n_vars, env)
    run_ea_ga_sa(toolbox, experiment_name, env)


def calculate_n_vars(env):
    """Calculate the number of variables for the neural network."""
    return (env.get_num_sensors() + 1) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5


def test_best_solution(enemy, i, test_experiment, env, best_solution_path, file_suffix):
    """Test the saved best solution."""
    try:
        best_solution = np.loadtxt(best_solution_path)
    except IOError:
        print(f"Error: Best solution for enemy {enemy} not found at {best_solution_path}.")
        return

    print(f'\nRUNNING SAVED BEST SOLUTION {file_suffix.upper()}\n')
    individual_gains = []

    with open(f'{test_experiment}/results_test{file_suffix}.csv', 'w', newline='') as file_aux:
        csv_writer = csv.writer(file_aux)
        csv_writer.writerow(['Run', 'Individual Gain'])

        for run in range(TEST_RUNS):
            print(f"Test run {run + 1} for enemy {enemy}")
            fitness, player_life, enemy_life, time = env.play(best_solution)
            individual_gain = player_life - enemy_life
            individual_gains.append(individual_gain)
            print(f"Individual gain for run {run + 1}: {individual_gain}")
            csv_writer.writerow([run + 1, individual_gain])

        avg_gain = np.mean(individual_gains)
        std_gain = np.std(individual_gains)
        print(f"\nAverage individual gain for enemy {enemy} over {TEST_RUNS} runs: {avg_gain}")
        print(f"Standard deviation: {std_gain}")

        csv_writer.writerow([])
        csv_writer.writerow(['Average Individual Gain', avg_gain])
        csv_writer.writerow(['Standard Deviation', std_gain])


def test_ga(enemy, i):
    """Test the Genetic Algorithm (GA) and GA+SA solutions."""
    test_experiment = f'test_run_enemy{enemy}/best_{i}'
    path_train = f'train_run_enemy{enemy}'
    if not os.path.exists(test_experiment):
        os.makedirs(test_experiment)

    env = create_environment(N_HIDDEN_NEURONS, test_experiment, enemy)

    # Test GA best solution
    best_solution_path_ga = f'{path_train}/GA_train_run{i}_enemy{enemy}/best.txt'
    test_best_solution(enemy, i, test_experiment, env, best_solution_path_ga, '')

    # Test GA+SA best solution
    best_solution_path_sa = f'{path_train}/GA_SA_train_run{i}_enemy{enemy}/best_sa.txt'
    test_best_solution(enemy, i, test_experiment, env, best_solution_path_sa, '_sa')


if __name__ == "__main__":
    if MODE == MODE_TRAIN:
        for enemy in ENEMIES:
            print("Running EA1_GA...")
            with mp.Pool(processes=os.cpu_count()) as pool:
                pool.starmap(train_ga, [(i, enemy) for i in range(TRAIN_RUNS)])

            print("Running EA2_GA_SA...")
            with mp.Pool(processes=os.cpu_count()) as pool:
                pool.starmap(train_ga_sa, [(i, enemy) for i in range(TRAIN_RUNS)])

    elif MODE == MODE_TEST:
        for enemy in ENEMIES:
            for i in range(1, TRAIN_RUNS + 1):
                print(f"Testing enemy {enemy}, run {i}")
                test_ga(enemy, i)
