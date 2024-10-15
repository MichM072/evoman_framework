# Importing necessary libraries
import sys

from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import os
from deap import base, creator, tools, algorithms
import multiprocessing as mp

# Set headless mode to speed up simulations
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Define constants
N_HIDDEN_NEURONS = 10
N_POPULATION = 100
N_GENERATIONS = 30
MUTATION_PROBABILITY = 0.1
CROSSOVER_PROBABILITY = 0.9
COOLING_RATE = 0.05  # For simulated annealing

ENEMIES = [2,4,8]
TRAIN_RUNS = 10
TEST_RUNS = 5
MODE_TRAIN = 'Train'
MODE_TEST = 'Test'

MODE = 'Train' # parameter can be Train or Test

# Define the environment setup
def get_environment(
        n_hidden_neurons: int,
        experiment_name: str,
        enemy: int
) -> Environment:
    return Environment(experiment_name=experiment_name,
                       enemies=[enemy],
                       playermode="ai",
                       player_controller=player_controller(n_hidden_neurons),
                       enemymode="static",
                       level=2,
                       speed="fastest",
                       visuals=False,
                       randomini='yes')

# Setup DEAP toolbox
def setup_deap(n_vars: int, env: Environment) -> base.Toolbox:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, -1, 1)  # Corrected order of limits
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, env)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

# Define evaluation function using the environment simulation
def evaluate(env, individual):
    # Return a single value instead of a tuple
    return simulation(env, individual),

# Simulate the environment and evaluate fitness
def simulation(env, x):
    f,_,_,_ = env.play(pcont=x)
    return f

# Custom similarity function for Hall of Fame
def custom_similar(ind1, ind2):
    # Compares individuals based on their array values
    return np.array_equal(ind1, ind2)

# This is implemented from the package DEAP following their documentation
def EA1_GA(
        toolbox: base.Toolbox,
        experiment_name: str,
        env: Environment,
):
    # Initialize population
    pop = toolbox.population(n=N_POPULATION)
    hof = tools.HallOfFame(1, similar=custom_similar)

    # Statistics to monitor the progress
    stats = tools.Statistics(lambda individual: individual.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the genetic algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CROSSOVER_PROBABILITY, mutpb=MUTATION_PROBABILITY, ngen=N_GENERATIONS,
                                   stats=stats, halloffame=hof, verbose=True)

    # Save the best solution
    np.savetxt(experiment_name + '/best.txt', hof[0])

    # Log results to a file
    with open(experiment_name + '/results_' + experiment_name.split("/")[-1] + '.txt', 'a') as file_aux:
        for gen in log:
            file_aux.write(f"Generation {gen['gen']}: Max {gen['max']} Avg {gen['avg']} Std {gen['std']}\n")

    # Save simulation state
    env.update_solutions([pop, [ind.fitness.values[0] for ind in pop]])
    env.save_state()

    print(f"Best individual with SA is: {hof[0]}")

# This is an adaptation from DEAP's implementation with a different mutation probability
def EA2_GA_SA(
        toolbox: base.Toolbox,
        experiment_name: str,
        env: Environment,
):
    # Initialize population
    pop = toolbox.population(n=N_POPULATION)

    hof = tools.HallOfFame(1, similar=custom_similar)

    # Statistics to monitor the progress
    stats = tools.Statistics(lambda individual: individual.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    log_file_path = experiment_name + '/results_' + experiment_name.split("/")[-1] + '.txt'

    # Open the log file in append mode
    with open(log_file_path, 'a') as log_file:
        # Run the genetic algorithm with SA-adjusted mutation probability
        for gen in range(N_GENERATIONS+1):
            # Calculate the dynamic mutation probability using a cooling schedule

            # Apply genetic algorithm steps manually to include custom mutation probability
            dynamic_mutpb = MUTATION_PROBABILITY * np.exp(-COOLING_RATE * gen)

            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < CROSSOVER_PROBABILITY:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.rand() < dynamic_mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update population and statistics
            pop[:] = offspring

            hof.update(pop)

            record = stats.compile(pop)

            # Print and log the generation statistics
            print(f"Gen {gen}: {record}")
            log_file.write(f"Generation {gen}: Max {record['max']} Avg {record['avg']} Std {record['std']}\n")

    np.savetxt(experiment_name + '/best_sa.txt', hof[0])

    print(f"Best individual with SA is: {hof[0]}")
    print(f"Best fitness with SA is: {hof[0].fitness.values[0]}")

    env.update_solutions([pop, [ind.fitness.values[0] for ind in pop]])
    env.save_state()

def train_EA1_GA(i: int, enemy: int):
    ga_experiment = 'train_run_enemy' + str(enemy)  + '/GA_train_run' + str(i + 1) + '_enemy' + str(enemy)
    if not os.path.exists(ga_experiment):
            os.makedirs(ga_experiment)

    env = get_environment(N_HIDDEN_NEURONS, ga_experiment, enemy=enemy)
    n_vars = (env.get_num_sensors() + 1) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5
    toolbox = setup_deap(n_vars, env=env)
    EA1_GA(toolbox, ga_experiment, env=env)

def train_EA2_GA_SA(i: int, enemy:int):
    ga_sa_experiment = 'train_run_enemy' + str(enemy) + '/GA_SA_train_run' + str(i + 1) + '_enemy' + str(enemy)
    if not os.path.exists(ga_sa_experiment):
        os.makedirs(ga_sa_experiment)

    env = get_environment(N_HIDDEN_NEURONS, ga_sa_experiment, enemy=enemy)
    n_vars = (env.get_num_sensors() + 1) * N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS + 1) * 5
    toolbox = setup_deap(n_vars, env=env)
    EA2_GA_SA(toolbox, ga_sa_experiment, env=env)

if __name__ == "__main__":
    if MODE == MODE_TRAIN:
        for enemy in ENEMIES:
            print("Running EA1_GA...")
            with mp.Pool(processes=os.cpu_count()) as pool:
                pool.starmap(train_EA1_GA, [(i, enemy) for i in range(TRAIN_RUNS)])

            print("Running EA2_GA_SA...")
            with mp.Pool(processes=os.cpu_count()) as pool:
                pool.starmap(train_EA2_GA_SA, [(i, enemy) for i in range(TRAIN_RUNS)])

    if MODE == MODE_TEST:
        for enemy in ENEMIES:
            for i in range(1, TRAIN_RUNS + 1):

                print(f"Testing enemy {enemy}")

                # Initialize the test experiment folder for each enemy
                test_experiment = f'test_run_enemy{enemy}/best_{i}'
                path_train = f'train_run_enemy{enemy}'
                if not os.path.exists(test_experiment):
                    os.makedirs(test_experiment)

                # Set up the environment for testing
                env = get_environment(N_HIDDEN_NEURONS, test_experiment, enemy=enemy)

                # Load the best solution obtained from the training phase (best.txt)
                try:
                    bsol = np.loadtxt(f'{path_train}/GA_train_run{i}_enemy{enemy}/best.txt')  # Adjust path as needed
                except IOError:
                    print(f"Error: Best solution for enemy {enemy} not found.")
                    continue

                print('\n RUNNING SAVED BEST SOLUTION \n')
                # env.update_parameter('speed', 'normal')

                individual_gains = []
                with open(f'{test_experiment}/results_test.txt', 'a') as file_aux:
                    file_aux.write(f"Results for Enemy {enemy} using best.txt:\n")
                    for run in range(TEST_RUNS):
                        print(f"Test run {run + 1} for enemy {enemy}")
                        fitness, player_life, enemy_life, time = env.play(bsol)
                        individual_gain = player_life - enemy_life
                        individual_gains.append(individual_gain)
                        print(f"Individual gain for run {run + 1}: {individual_gain}")
                        # Log individual gain for each run
                        file_aux.write(f"Run {run + 1}: Individual Gain = {individual_gain}\n")

                    # Calculate and display the average performance
                    avg_gain = np.mean(individual_gains)
                    std_gain = np.std(individual_gains)
                    print(f"\nAverage individual gain for enemy {enemy} over {TEST_RUNS} runs: {avg_gain}")
                    print(f"Standard deviation: {std_gain}")

                    # Log average and standard deviation to the file
                    file_aux.write(f"Average Individual Gain: {avg_gain}\n")
                    file_aux.write(f"Standard Deviation: {std_gain}\n\n")

                # Load the best solution obtained from the training phase (best_sa.txt)
                try:
                    bsol_sa = np.loadtxt(f'{path_train}/GA_SA_train_run{i}_enemy{enemy}/best_sa.txt')  # Adjust path as needed
                except IOError:
                    print(f"Error: Best solution with simulated annealing for enemy {enemy} not found.")
                    continue

                print('\n RUNNING SAVED BEST SOLUTION WITH SIMULATED ANNEALING \n')
                # env.update_parameter('speed', 'normal')

                individual_gains_sa = []
                with open(f'{test_experiment}/results_test_sa.txt', 'a') as file_aux_sa:
                    file_aux_sa.write(f"Results for Enemy {enemy} using best_sa.txt:\n")
                    for run in range(TEST_RUNS):
                        print(f"Test run {run + 1} for enemy {enemy} using best_sa.txt")
                        fitness, player_life, enemy_life, time = env.play(bsol_sa)
                        individual_gain_sa = player_life - enemy_life
                        individual_gains_sa.append(individual_gain_sa)
                        print(f"Individual gain for run {run + 1}: {individual_gain_sa}")
                        # Log individual gain for each run
                        file_aux_sa.write(f"Run {run + 1}: Individual Gain = {individual_gain_sa}\n")

                    # Calculate and display the average performance
                    avg_gain_sa = np.mean(individual_gains_sa)
                    std_gain_sa = np.std(individual_gains_sa)
                    print(f"\nAverage individual gain for enemy {enemy} over {TEST_RUNS} runs using best_sa.txt: {avg_gain_sa}")
                    print(f"Standard deviation: {std_gain_sa}")

                    # Log average and standard deviation to the file
                    file_aux_sa.write(f"Average Individual Gain: {avg_gain_sa}\n")
                    file_aux_sa.write(f"Standard Deviation: {std_gain_sa}\n\n")
