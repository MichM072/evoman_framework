import os
import csv
import numpy as np
import multiprocessing as mp
from deap import base, creator, tools, algorithms
from evoman.environment import Environment
from demo_controller import player_controller
import statistics

os.environ["SDL_VIDEODRIVER"] = "dummy"

# Constants and configurations
N_HIDDEN_NEURONS = 10
N_POPULATION = 100
N_GENERATIONS = 30
MUTATION_PROBABILITY = 0.05
CROSSOVER_PROBABILITY = 0.9
NUM_GENERATIONS_WITHOUT_GROWTH = 3  # Threshold for stagnation
ELITISM_RATE = 0.05  # Percentage of best to keep
N_LEAST_PERFORMING = 5  # Remove num of ind that are performing very bad
ENEMY_GROUPS = [[1,4,7,6], [1,8,3,7,6,5]] #TODO: Change default values
TRAIN_RUNS = 10
TEST_RUNS = 10
SIGNIFICANT_GROWTH = 1

UP_LIMIT = 1
LOWER_LIMIT = -1

GEN_THRESHOLD = 5

MODE_TRAIN = 'Train'
MODE_TEST = 'Test'
MODE = MODE_TEST  #'Train' or 'Test' based on your needs

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
def check_significant_growth(group, threshold=SIGNIFICANT_GROWTH):
    sum_all_ind = np.sum([(ind.sum_growth / GEN_THRESHOLD) for ind in group])
    return (sum_all_ind / len(group)) >= threshold

#Get sum of all gens for individual and divide by gen threshold
def check_individual_significant_growth(ind, threshold=SIGNIFICANT_GROWTH):
    return (ind.sum_growth / GEN_THRESHOLD) >= threshold

def check_growth(ind):
   ind.sum_growth += ind.fitness.values[0] - ind.prev_fitness
   ind.prev_fitness = ind.fitness.values[0]

def increase_mutation_rate(mutation_rate):
    return min(1, mutation_rate * 1.5) #TOOD: Play around with these values to see any meaningful effect.

def decrease_mutation_rate(mutation_rate):
    return max(0.005, mutation_rate * 0.5)  #TOOD: Play around with these values to see any meaningful effect.

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
    #AGE-BASED APPROACH
    # Sort by fitness (good to bad)
    num_elite_by_fitness = int(len(group) * elitism_rate)
    sorted_by_fitness = sorted(group, key=lambda x: x.fitness.values[0], reverse=True)
    top_by_fitness = sorted_by_fitness[:num_elite_by_fitness]

    # Sort by age and select top 10% oldest individuals
    # num_elite_by_age = max(1, int(len(group) * 0.10))  # Ensure at least one individual is selected
    # sorted_by_age = sorted(group, key=lambda x: x.age, reverse=True)
    # top_by_age = [ind for ind in sorted_by_age if ind.age >= age_threshold][:num_elite_by_age]

    elites = list(top_by_fitness)

    return elites

#TODO: Perhaps replace this with a singular elite?
# e.g. only keep the best for each gen.
def replace_with_elites(elites, offspring):
    # Ensure you do not replace your own elite.
    offset = 0
    for i in range(len(elites)):
        for j in range(offset, len(offspring)):
            if elites[i].fitness.values[0] > offspring[j].fitness.values[0]:
                # If elite is better than poor performer replace.
                offspring[j] = elites[i]
                assert offspring[j].fitness.values[0] == elites[i].fitness.values[0]
                offset = j + 1
                break
            elif elites[i].fitness.values[0] < offspring[j].fitness.values[0]:
                # Stop if the elite has a lower value than the item in the offspring.
                # This allows us to continue if it is equal.
                break

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
                      mutation_rate_A, mutation_rate_B):
    invalid_individuals_A = [ind for ind in Group_A if not ind.fitness.valid]
    invalid_individuals_B = [ind for ind in Group_B if not ind.fitness.valid]


    if invalid_individuals_A:
        evaluate_invalid_individuals(toolbox, invalid_individuals_A)

    if invalid_individuals_B:
        evaluate_invalid_individuals(toolbox, invalid_individuals_B)

    # Perform age-based elitism with a 10% threshold
    # TODO: what is the age_threshold???
    elites_A = elitism(Group_A, ELITISM_RATE) if Group_A else []
    elites_B = elitism(Group_B, ELITISM_RATE) if Group_B else []

    offspring_A = toolbox.select(Group_A, len(Group_A))
    offspring_B = toolbox.select(Group_B, len(Group_B))

    offspring_A = list(map(toolbox.clone, offspring_A))
    offspring_B = list(map(toolbox.clone, offspring_B))

    crossover_and_mutate(offspring_A, toolbox, mutation_rate_A)
    crossover_and_mutate(offspring_B, toolbox, mutation_rate_B)

    # Fill population back to original size after elitism
    offspring_A = sorted(offspring_A, key=lambda x: x.fitness.values[0], reverse=False)
    offspring_B = sorted(offspring_B, key=lambda x: x.fitness.values[0], reverse=False)

    # Make sure the lite list is also sorted from lowest elite to highest.
    elites_A.sort(key=lambda x: x.fitness.values[0], reverse=False)
    elites_B.sort(key=lambda x: x.fitness.values[0], reverse=False)

    # Replace the worst performers with elites.
    # If the elite is not better than any of the individuals it is dropped.
    for elites, offspring in zip([elites_A, elites_B], [offspring_A, offspring_B]):
        replace_with_elites(elites, offspring)

    # Swap between groups if there is (no) significant growth.
    # Move to A from B if there is growth and vice versa.
    for ind in offspring_A:
        if ind.gen == GEN_THRESHOLD:
            ind.gen = 0
            if not check_individual_significant_growth(ind, SIGNIFICANT_GROWTH):
                # print(f"Moving individual with growth: {ind.sum_growth/5} and fitness:{ind.fitness.values[0]} to group B")
                move_to_group_B(ind, offspring_A, offspring_B)
            ind.sum_growth = 0

    for ind in offspring_B:
        if ind.gen == GEN_THRESHOLD:
            ind.gen = 0
            if check_individual_significant_growth(ind, SIGNIFICANT_GROWTH):
                # print(f"Moving individual with growth: {ind.sum_growth / 5} and fitness:{ind.fitness.values[0]} to group A")
                move_to_group_B(ind, offspring_B, offspring_A)
            ind.sum_growth = 0


    # Increase gen counter per individual after elitism
    for ind in offspring_A + offspring_B:
        check_growth(ind)
        ind.gen += 1

    invalid_individuals = [ind for ind in offspring_A + offspring_B if not ind.fitness.valid]
    evaluate_invalid_individuals(toolbox, invalid_individuals)

    return offspring_A, offspring_B

def evolve_population_EA2(pop, toolbox, history_pop, mutation_rate, generation):
    invalid_individuals = [ind for ind in pop if not ind.fitness.valid]

    if invalid_individuals:
        fitnesses = map(toolbox.evaluate, invalid_individuals)
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit  # Assign the fitness values

    # Increase or decrease mutation rate based on growth
    if (generation + 1) % 5 == 0:
        if not check_significant_growth(pop) and pop:
            mutation_rate = increase_mutation_rate(mutation_rate)
        else:
            mutation_rate = decrease_mutation_rate(mutation_rate)

    if pop:  # Only perform crossover if population is not empty
        elites = elitism(pop, ELITISM_RATE) if pop else []
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

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
        replace_with_elites(elites, offspring)
        pop[:] = offspring

    # TODO: Add this back if we decide to use this survivor selection with Elitism
    # Keep in mind we should then adjust the elitism too!
    # pop = remove_least_performers(pop, N_LEAST_PERFORMING) if pop else pop

    for ind in pop:
        check_growth(ind)
        ind.gen += 1

    return pop


def train_ea1(i, enemies):
    experiment_name = f'train_run_enemy{enemies}/EA1_train_run{i + 1}_enemy{enemies}'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    env = create_environment(N_HIDDEN_NEURONS, experiment_name, enemies)
    n_vars = calculate_n_vars(env)
    toolbox = initialize_deap_toolbox(n_vars, env)

    # Initialize A and B
    Group_A = toolbox.population(n=N_POPULATION)
    Group_B = []

    mutation_rate_A = MUTATION_PROBABILITY
    mutation_rate_B = 0.3 # High mutation rate to force stagnant individuals to explore.
    history_A, history_B = [], []


    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    log_file_path = f"{experiment_name}/results_EA1.csv"

    with open(log_file_path, 'w', newline='') as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(['Generation', 'Max', 'Avg', 'Std', 'Min'])

        for generation in range(N_GENERATIONS):
            Group_A, Group_B = evolve_population(Group_A, Group_B, toolbox, history_A, history_B,
                                                 mutation_rate_A, mutation_rate_B)

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

        # Choose the best solution between best_A and best_B and save it as best.txt
        if best_B > best_A:
            print(f"Best_B selected with fitness: {best_B}")
            save_best_solution(Group_B[0], experiment_name, 'best.txt')
        else:
            print(f"Best_A selected with fitness: {best_A}")
            save_best_solution(Group_A[0], experiment_name, 'best.txt')


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
            pop = evolve_population_EA2(pop, toolbox, history, mutation_rate, generation)

            # Are individuals valid? Check!
            best = max(ind.fitness.values[0] for ind in pop if ind.fitness.valid)

            history.append(best)

            print(f"Generation {generation}: Best: {best}")

            record = stats.compile(pop)
            log_generation_to_csv(csv_writer, generation, record)

        save_best_solution(pop[0], experiment_name, 'best.txt')


def test_best_solution(run_number, test_experiment, env, best_solution_path, file_suffix):
    try:
        best_solution = np.loadtxt(best_solution_path)
    except IOError:
        print(f"ERROR: Best solution for run {run_number} not found at {best_solution_path}.")
        return

    log_file_path_gain = os.path.join(test_experiment, f'results_test_{file_suffix}.csv')

    fitness, player_life, enemy_life, time = env.play(best_solution)
    individual_gain = player_life - enemy_life

    with open(log_file_path_gain, 'a', newline='') as file_aux:
        csv_writer = csv.writer(file_aux)
        csv_writer.writerow([run_number, individual_gain])

    return individual_gain


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

def save_results_to_csv(gains_list, file_path):
    # Calculate average and standard deviation
    gains = [item['Gain'] for item in gains_list]
    average_gain = sum(gains) / len(gains)
    std_dev = statistics.stdev(gains) if len(gains) > 1 else 0.0

    # Write to CSV
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['Run', 'Gain']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for item in gains_list:
            writer.writerow(item)

        # Write average and standard deviation
        writer.writerow({})
        writer.writerow({'Run': 'Average Individual Gain', 'Gain': average_gain})
        writer.writerow({'Run': 'Standard Deviation', 'Gain': std_dev})

if __name__ == "__main__":
    if MODE == MODE_TRAIN:
        for enemy_group in ENEMY_GROUPS:
            print(f"Training EA1 for group: {enemy_group}...")

            #TODO: REMOVE THIS, slow but works on macbook.
            # for i in range(TRAIN_RUNS):
                # train_ea1(i, enemy_group)

            with mp.Pool(processes=os.cpu_count()) as pool:
                pool.starmap(train_ea1, [(i, enemy_group) for i in range(TRAIN_RUNS)])

            print(f"Training EA2 for group: {enemy_group}...")

            with mp.Pool(processes=os.cpu_count()) as pool:
                pool.starmap(train_ea2, [(i, enemy_group) for i in range(TRAIN_RUNS)])

    if MODE == MODE_TEST:
        for enemy_group in ENEMY_GROUPS:
            # Create the folder structure for the test (e.g., test_enemy[...])
            test_experiment_base = f'test_enemy{enemy_group}'
            if not os.path.exists(test_experiment_base):
                os.makedirs(test_experiment_base)

            gains_EA1 = []
            gains_EA2 = []

            env = create_environment(N_HIDDEN_NEURONS, test_experiment_base, [1,2,3,4,5,6,7,8])

            # Loop through 10 runs (TRAIN_RUNS)
            for run_number in range(1, TRAIN_RUNS + 1):
                # Paths to the best solution files for EA1 and EA2
                best_solution_path_EA1 = f'train_run_enemy{enemy_group}/EA1_train_run{run_number}_enemy{enemy_group}/best.txt'
                best_solution_path_EA2 = f'train_run_enemy{enemy_group}/EA2_train_run{run_number}_enemy{enemy_group}/best.txt'

                # Test EA1 Best Solution
                try:
                    gain_EA1 = test_best_solution(run_number, test_experiment_base, env, best_solution_path_EA1, 'EA1')
                    gains_EA1.append({'Run': run_number, 'Gain': gain_EA1})
                except IOError:
                    print(f"Error: Best solution for enemy group {enemy_group} in run {run_number} not found for EA1.")

                # Test EA2 Best Solution
                try:
                    gain_EA2 = test_best_solution(run_number, test_experiment_base, env, best_solution_path_EA2, 'EA2')
                    gains_EA2.append({'Run': run_number, 'Gain': gain_EA2})
                except IOError:
                    print(f"Error: Best solution for enemy group {enemy_group} in run {run_number} not found for EA2.")

            # Save results to CSV files for each algorithm
            save_results_to_csv(gains_EA1, os.path.join(test_experiment_base, 'results_test_EA1.csv'))
            save_results_to_csv(gains_EA2, os.path.join(test_experiment_base, 'results_test_EA2.csv'))

            with open(os.path.join(test_experiment_base, 'evoman_logs.txt'), 'w') as log_file:
                log_file.write(f'Log information for enemy group {enemy_group}...\n')
