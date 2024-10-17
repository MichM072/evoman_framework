import os
import csv
import numpy as np
import multiprocessing as mp
from deap import base, creator, tools, algorithms
from evoman.environment import Environment
from demo_controller import player_controller

os.environ["SDL_VIDEODRIVER"] = "dummy"

# Constants and configurations
N_HIDDEN_NEURONS = 10
N_POPULATION = 100
N_GENERATIONS = 30
MUTATION_PROBABILITY = 0.05
CROSSOVER_PROBABILITY = 0.9
NUM_GENERATIONS_WITHOUT_GROWTH = 5  # Threshold for stagnation
ELITISM_RATE = 0.2  # Percentage of best to keep
N_LEAST_PERFORMING = 5  # Remove num of ind that are performing very bad
ENEMY_GROUPS = [[2,5,6], [5,7,8]] #TODO: Change default values
TRAIN_RUNS = 10
TEST_RUNS = 5
SIGNIFICANT_GROWTH = 5

UP_LIMIT = 1
LOWER_LIMIT = -1

GEN_THRESHOLD = 5

MODE_TRAIN = 'Train'
MODE_TEST = 'Test'
MODE = MODE_TRAIN  #'Train' or 'Test' based on your needs

Group_A = []
Group_B = []


def create_environment(n_hidden_neurons, experiment_name, enemies):
    return Environment(
        experiment_name=experiment_name,
        enemies=enemies,
        multiplemode="yes", # THIS MUST REMAIN YES
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest",
        visuals=False,
        randomini='no'
    )


def initialize_deap_toolbox(n_vars, env):
    # Avoid annoying warning
    if 'FitnessMax' not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    if 'Individual' not in creator.__dict__:
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax, id=None,
                       gen=0, sum_growth=0, prev_fitness=0)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual, env)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

#TODO: Apply normalisation in the code.
def normalize_value(value, data_population):
    value_range = np.ptp(data_population)  # using numpy for simplicity (max-min)
    normalized_value = (value - np.min(data_population)) / value_range if value_range > 0 else 0

    return max(normalized_value, 1e-10)

# Limits the range of all floats in individual between upper and lower limit.
def limit_range(ind):
    ind[ind > UP_LIMIT] = UP_LIMIT
    ind[ind < LOWER_LIMIT] = LOWER_LIMIT


def evaluate_individual(env, individual):
    return simulate_environment(env, individual),

def evaluate_invalid_individuals(toolbox, invalid_individuals):
    fitnesses = map(toolbox.evaluate, invalid_individuals)
    for ind, fit in zip(invalid_individuals, fitnesses):
        ind.fitness.values = fit  # Assign the fitness values


def simulate_environment(env, individual):
    fitness, _, _, _ = env.play(pcont=individual)
    return fitness


#TODO: Check growth such that it is the growth of the whole group
def check_significant_growth(group, history, threshold=NUM_GENERATIONS_WITHOUT_GROWTH):
    if len(history) < threshold:
        return True  #Cannot judge if not enough info

    return np.mean(history[-threshold:]) > np.mean(history[-2 * threshold:-threshold]) # TODO: make this more readable pls

#TODO: Adjust function to check growth of individuals
#Get sum of all 5 gens for individual and divide by 5
def check_individual_significant_growth(ind, threshold=SIGNIFICANT_GROWTH):
    return (ind.sum_growth / GEN_THRESHOLD) > threshold

def check_growth(ind):
   ind.sum_growth += ind.fitness.values[0] - ind.prev_fitness
   ind.prev_fitness = ind.fitness.values[0]

def increase_mutation_rate(mutation_rate):
    return min(1.0, mutation_rate * 1.1)  ## TODO: What is the min? Is 1.0 ok?

#TODO: We do not crossover between groups.
def crossover_and_mutate(group, toolbox, mutation_rate):
    if not group:
        return
    for ind1, ind2 in zip(group[::2], group[1::2]):
        if np.random.rand() < CROSSOVER_PROBABILITY:
            toolbox.mate(ind1, ind2)
            del ind1.fitness.values, ind2.fitness.values
    for ind in group:
        if np.random.rand() < mutation_rate:
            toolbox.mutate(ind)
            del ind.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in group if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        limit_range(ind)
        ind.fitness.values = fit


def elitism(group, elitism_rate):
    num_elite = int(len(group) * elitism_rate)
    if num_elite == 0:
        return group
    sorted_group = sorted(group, key=lambda x: x.fitness.values[0], reverse=True)
    return sorted_group[:num_elite]


def get_fitness_value(individual):
    return individual.fitness.values[0]

def remove_least_performers(group, num_individuals_to_remove):
    sorted_group = sorted(group, key=get_fitness_value)
    return sorted_group[num_individuals_to_remove:]


def move_to_group_B(individual, Group_A, Group_B):
    # Group_A = [ind for ind in Group_A if not np.array_equal(ind, individual)]
    Group_B.append(individual)
    Group_A = [arr for arr in Group_A if not np.array_equal(arr, individual)]


def move_to_group_A(individual, Group_A, Group_B):
    # Group_B = [ind for ind in Group_B if not np.array_equal(ind, individual)]
    Group_A.append(individual)
    Group_B = [arr for arr in Group_B if not np.array_equal(arr, individual)]


def evolve_population(Group_A, Group_B, toolbox, history_A, history_B,
                      mutation_rate_A, mutation_rate_B, ind_dict):
    invalid_individuals_A = [ind for ind in Group_A if not ind.fitness.valid]
    invalid_individuals_B = [ind for ind in Group_B if not ind.fitness.valid]

    if invalid_individuals_A:
        evaluate_invalid_individuals(toolbox, invalid_individuals_A)

    if invalid_individuals_B:
        evaluate_invalid_individuals(toolbox, invalid_individuals_B)

    crossover_and_mutate(Group_A, toolbox, mutation_rate_A)
    crossover_and_mutate(Group_B, toolbox, mutation_rate_B)

#TODO: We need to check the individuals on significant growth, not on fitness.
    for ind in Group_A:
        if ind.gen == GEN_THRESHOLD:
            ind.gen = 0
            if not check_individual_significant_growth(ind, SIGNIFICANT_GROWTH):
                move_to_group_B(ind, Group_A, Group_B)
            ind.sum_growth = 0

    for ind in Group_B:
        if ind.gen == GEN_THRESHOLD:
            ind.gen = 0
            if check_individual_significant_growth(ind, SIGNIFICANT_GROWTH):
                move_to_group_B(ind, Group_B, Group_A)
            ind.sum_growth = 0
    # if Group_A:  # Ensure Group_A is not empty
    #     avg_fitness_A = np.mean([i.fitness.values[0] for i in Group_A]) if Group_A else 0
    #     for ind in Group_A:
    #         if ind.fitness.values[0] < avg_fitness_A:
    #             move_to_group_B(ind, Group_A, Group_B)
    #
    # if Group_B:  # Ensure Group_B is not empty
    #     avg_fitness_B = np.mean([i.fitness.values[0] for i in Group_B]) if Group_B else 0
    #     for ind in Group_B:
    #         if ind.fitness.values[0] > avg_fitness_B:
    #             move_to_group_A(ind, Group_A, Group_B)

    original_size_A = len(Group_A)
    original_size_B = len(Group_B)

    Group_A = elitism(Group_A, ELITISM_RATE) if Group_A else Group_A
    Group_B = elitism(Group_B, ELITISM_RATE) if Group_B else Group_B

    # Group_A = remove_least_performers(Group_A, N_LEAST_PERFORMING) if Group_A else Group_A
    # Group_B = remove_least_performers(Group_B, N_LEAST_PERFORMING) if Group_B else Group_B

    # Reproduce
    # TODO: Check if this logic holds, we flood group_B with individuals that might not even belong there.
    # Add only the amount of individuals that we removed from said group.

    # Increase gen counter per individual after elitism
    for ind in Group_A + Group_B:
        check_growth(ind)
        ind.gen += 1

    if Group_A:
        Group_A += toolbox.population(n=original_size_A - len(Group_A))
    if Group_B:
        Group_B += toolbox.population(n=original_size_B - len(Group_B))

    invalid_individuals = [ind for ind in Group_A + Group_B if not ind.fitness.valid]
    evaluate_invalid_individuals(toolbox, invalid_individuals)

    return Group_A, Group_B

def evolve_population_EA2(pop, toolbox, history_pop, mutation_rate):
    invalid_individuals = [ind for ind in pop if not ind.fitness.valid]

    if invalid_individuals:
        fitnesses = map(toolbox.evaluate, invalid_individuals)
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit  # Assign the fitness values

#TODO: This growth check should be different, we should discuss with Luiz and Georgia.
    if not check_significant_growth(pop, history_pop) and pop:
        mutation_rate = increase_mutation_rate(mutation_rate)

    if pop:  # Only perform crossover if population is not empty
        offspring = list(map(toolbox.clone, pop))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < CROSSOVER_PROBABILITY:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.rand() < mutation_rate:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population with offspring
        pop[:] = offspring

    pop = elitism(pop, ELITISM_RATE) if pop else pop

    # pop = remove_least_performers(pop, N_LEAST_PERFORMING) if pop else pop

    # Reproduce
    if pop:
        pop += toolbox.population(n=N_POPULATION - len(pop))

    return pop


def train_ea1(i, enemies):
    experiment_name = f'train_run_enemy{enemies}/EA1_train_run{i + 1}_enemy{enemies}'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    env = create_environment(N_HIDDEN_NEURONS, experiment_name, enemies)
    n_vars = calculate_n_vars(env)
    toolbox = initialize_deap_toolbox(n_vars, env)

    # Create dict to track individuals and gen
    # e.g. individual 1 with id 1 with 2 gens = [1,2]
    ind_dict = {}

    # Initialize A and B
    Group_A = toolbox.population(n=N_POPULATION)
    Group_B = []

    # Assigns id to individual and add to dict.
    # for idx, ind in enumerate(Group_A):
    #     ind.id = idx
    #     ind_dict[ind.id] = 0

    mutation_rate_A = MUTATION_PROBABILITY
    mutation_rate_B = 0.1
    history_A, history_B = [], []


    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    log_file_path = f"{experiment_name}/results_EA2.csv"

    with open(log_file_path, 'w', newline='') as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(['Generation', 'Max', 'Avg', 'Std', 'Min'])

        for generation in range(N_GENERATIONS):
            Group_A, Group_B = evolve_population(Group_A, Group_B, toolbox, history_A, history_B,
                                                 mutation_rate_A, mutation_rate_B, ind_dict)

            # Are individuals valid? Check!
            best_A = max(ind.fitness.values[0] for ind in Group_A if ind.fitness.valid)
            if Group_B:
                best_B = max(ind.fitness.values[0] for ind in Group_B if ind.fitness.valid)
            else:
                best_B = 0

            history_A.append(best_A)
            history_B.append(best_B)

            print(f"Generation {generation}: Best A: {best_A}, Best B: {best_B}")

            for ind in Group_A:
                if not ind.fitness.valid:
                    print("invalid fitness found!")

            record = stats.compile(Group_A)
            log_generation_to_csv(csv_writer, generation, record)

    save_best_solution(Group_A[0], experiment_name, 'best_A.txt')
    save_best_solution(Group_B[0], experiment_name, 'best_B.txt')


def train_ea2(i, enemies):
    experiment_name = f'train_run_enemy{enemies}/EA2_train_run{i + 1}_enemy{enemies}'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    env = create_environment(N_HIDDEN_NEURONS, experiment_name, enemies)
    n_vars = calculate_n_vars(env)
    toolbox = initialize_deap_toolbox(n_vars, env)

    # Initialize population
    pop = toolbox.population(n=N_POPULATION)

    mutation_rate = MUTATION_PROBABILITY
    history = []

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    log_file_path = f"{experiment_name}/results_EA2.csv"

    with open(log_file_path, 'w', newline='') as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(['Generation', 'Max', 'Avg', 'Std', 'Min'])

        for generation in range(N_GENERATIONS):
            pop = evolve_population_EA2(pop, toolbox, history, mutation_rate,)

            # Are individuals valid? Check!
            best = max(ind.fitness.values[0] for ind in pop if ind.fitness.valid)

            history.append(best)

            print(f"Generation {generation}: Best: {best}")

            record = stats.compile(pop)
            log_generation_to_csv(csv_writer, generation, record)

        save_best_solution(pop[0], experiment_name, 'best_A.txt')

def test_best_solution(enemy, i, test_experiment, env, best_solution_path, file_suffix):
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

def calculate_n_vars(env):
    return (env.get_num_sensors() + 1) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5

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

if __name__ == "__main__":
    if MODE == MODE_TRAIN:
        for enemy_group in ENEMY_GROUPS:
            print(f"Training EA1 for group: {enemy_group}...")

            #TODO: REMOVE THIS, slow but works on macbook.
            for i in range(TRAIN_RUNS):
                train_ea1(i, enemy_group)

            # with mp.Pool(processes=os.cpu_count()) as pool:
            #     pool.starmap(train_ea1, [(i, enemy_group) for i in range(TRAIN_RUNS)])

            # print(f"Training EA2 for group: {enemy_group}...")
            #
            # with mp.Pool(processes=os.cpu_count()) as pool:
            #     pool.starmap(train_ea2, [(i, enemy_group) for i in range(TRAIN_RUNS)])

    elif MODE == MODE_TEST:
        for enemy in ENEMY_GROUPS:
            for i in range(1, TRAIN_RUNS + 1):

                print(f"Testing enemy {enemy}")

                test_experiment = f'test_run_enemy{enemy}/best_{i}'
                path_train = f'train_run_enemy{enemy}'
                if not os.path.exists(test_experiment):
                    os.makedirs(test_experiment)

                env = create_environment(N_HIDDEN_NEURONS, test_experiment, enemy)

                # Load the best solution
                try:
                    best_solution_path_ga = f'{path_train}/EA1_train_run{i}_enemy{enemy}/best_A.txt'
                    test_best_solution(enemy, i, test_experiment, env, best_solution_path_ga, '')

                except IOError:
                    print(f"Error: Best solution for enemy {enemy} not found.")
