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
MUTATION_PROBABILITY = 0.1
CROSSOVER_PROBABILITY = 0.9
NUM_GENERATIONS_WITHOUT_GROWTH = 5  # Threshold for stagnation
ELITISM_RATE = 0.2  # Percentage of best to keep
N_LEAST_PERFORMING = 5  # Remove num of ind that are performing very bad
ENEMIES = [2, 4, 8]
TRAIN_RUNS = 10
TEST_RUNS = 5

MODE_TRAIN = 'Train'
MODE_TEST = 'Test'
MODE = MODE_TRAIN  #'Train' or 'Test' based on your needs

Group_A = []
Group_B = []


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
    # Avoid annoying warning
    if 'FitnessMax' not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    if 'Individual' not in creator.__dict__:
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


def check_significant_growth(group, history, threshold=NUM_GENERATIONS_WITHOUT_GROWTH):
    if len(history) < threshold:
        return True  #Cannot judge if not enough info

    return np.mean(history[-threshold:]) > np.mean(history[-2 * threshold:-threshold]) # TODO: make this more readable pls


def increase_mutation_rate(mutation_rate):
    return min(1.0, mutation_rate * 1.1)  ## TODO: What is the min? Is 1.0 ok?


def crossover_and_mutate(group_1, group_2, toolbox, mutation_rate):
    for ind1, ind2 in zip(group_1[:len(group_1) // 2], group_2[:len(group_2) // 2]):
        toolbox.mate(ind1, ind2)
        toolbox.mutate(ind1)
        toolbox.mutate(ind2)


def elitism(group, elitism_rate):
    num_elite = int(len(group) * elitism_rate)
    sorted_group = sorted(group, key=lambda x: x.fitness.values[0], reverse=True)
    return sorted_group[:num_elite]


def remove_least_performers(group, num_individuals_to_remove):
    sorted_group = sorted(group, key=lambda x: x.fitness.values[0])
    return sorted_group[num_individuals_to_remove:]


def move_to_group_B(individual, Group_A, Group_B):
    # Group_A = [ind for ind in Group_A if not np.array_equal(ind, individual)]
    Group_B.append(individual)


def move_to_group_A(individual, Group_A, Group_B):
    # Group_B = [ind for ind in Group_B if not np.array_equal(ind, individual)]
    Group_A.append(individual)


def evolve_population(Group_A, Group_B, toolbox, history_A, history_B, mutation_rate_A, mutation_rate_B):
    invalid_individuals_A = [ind for ind in Group_A if not ind.fitness.valid]
    invalid_individuals_B = [ind for ind in Group_B if not ind.fitness.valid]

    if invalid_individuals_A:
        fitnesses_A = map(toolbox.evaluate, invalid_individuals_A)
        for ind, fit in zip(invalid_individuals_A, fitnesses_A):
            ind.fitness.values = fit  # Assign the fitness values

    if invalid_individuals_B:
        fitnesses_B = map(toolbox.evaluate, invalid_individuals_B)
        for ind, fit in zip(invalid_individuals_B, fitnesses_B):
            ind.fitness.values = fit  # Assign the fitness values

    if not check_significant_growth(Group_A, history_A) and Group_A:
        mutation_rate_A = increase_mutation_rate(mutation_rate_A)

    if not check_significant_growth(Group_B, history_B) and Group_B:
        mutation_rate_B = increase_mutation_rate(mutation_rate_B)

    if Group_A and Group_B:  # Only perform crossover if both groups are non-empty
        crossover_and_mutate(Group_A, Group_B, toolbox, mutation_rate_A)
        crossover_and_mutate(Group_B, Group_A, toolbox, mutation_rate_B)

    if Group_A:  # Ensure Group_A is not empty
        avg_fitness_A = np.mean([i.fitness.values[0] for i in Group_A]) if Group_A else 0
        for ind in Group_A:
            if ind.fitness.values[0] < avg_fitness_A:
                move_to_group_B(ind, Group_A, Group_B)

    if Group_B:  # Ensure Group_B is not empty
        avg_fitness_B = np.mean([i.fitness.values[0] for i in Group_B]) if Group_B else 0
        for ind in Group_B:
            if ind.fitness.values[0] > avg_fitness_B:
                move_to_group_A(ind, Group_A, Group_B)

    Group_A = elitism(Group_A, ELITISM_RATE) if Group_A else Group_A
    Group_B = elitism(Group_B, ELITISM_RATE) if Group_B else Group_B

    Group_A = remove_least_performers(Group_A, N_LEAST_PERFORMING) if Group_A else Group_A
    Group_B = remove_least_performers(Group_B, N_LEAST_PERFORMING) if Group_B else Group_B

    # Reproduce
    if Group_A:
        Group_A += toolbox.population(n=N_POPULATION - len(Group_A))
    if Group_B:
        Group_B += toolbox.population(n=N_POPULATION - len(Group_B))

    return Group_A, Group_B


def train_ea1(i, enemy):
    experiment_name = f'train_run_enemy{enemy}/EA1_train_run{i + 1}_enemy{enemy}'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    env = create_environment(N_HIDDEN_NEURONS, experiment_name, enemy)
    n_vars = calculate_n_vars(env)
    toolbox = initialize_deap_toolbox(n_vars, env)

    # Initialize A and B
    Group_A = toolbox.population(n=N_POPULATION)
    Group_B = toolbox.population(n=N_POPULATION)

    mutation_rate_A = MUTATION_PROBABILITY
    mutation_rate_B = MUTATION_PROBABILITY
    history_A, history_B = [], []

    for generation in range(N_GENERATIONS):
        Group_A, Group_B = evolve_population(Group_A, Group_B, toolbox, history_A, history_B, mutation_rate_A,
                                             mutation_rate_B)

        # Are individuals valid? Check!
        best_A = max(ind.fitness.values[0] for ind in Group_A if ind.fitness.valid)
        best_B = max(ind.fitness.values[0] for ind in Group_B if ind.fitness.valid)

        history_A.append(best_A)
        history_B.append(best_B)

        print(f"Generation {generation}: Best A: {best_A}, Best B: {best_B}")

    save_best_solution(Group_A[0], experiment_name, 'best_A.txt')
    save_best_solution(Group_B[0], experiment_name, 'best_B.txt')


# Placeholder for EA2 (not implemented yet)
def train_ea2(i, enemy):
    pass


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


def save_best_solution(solution, experiment_name, file_name):
    np.savetxt(f'{experiment_name}/{file_name}', solution)


def calculate_n_vars(env):
    return (env.get_num_sensors() + 1) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5


if __name__ == "__main__":
    if MODE == MODE_TRAIN:
        for enemy in ENEMIES:
            print(f"Training EA1 for Enemy {enemy}...")

            with mp.Pool(processes=os.cpu_count()) as pool:
                pool.starmap(train_ea1, [(i, enemy) for i in range(TRAIN_RUNS)])

            # print(f"Training EA2 for Enemy {enemy}...")

            # Placeholder for EA2; currently does nothing
            # with mp.Pool(processes=os.cpu_count()) as pool:
            #     pool.starmap(train_ea2, [(i, enemy) for i in range(TRAIN_RUNS)])

    elif MODE == MODE_TEST:
        for enemy in ENEMIES:
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
