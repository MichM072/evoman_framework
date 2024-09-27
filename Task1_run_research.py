###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# Test : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Group 55   			                                              #			                                  #
###############################################################################
import numpy as np

from GA_Spec_SA import GASpecialistSA
import matplotlib
import os
import pandas as pd

'''
Steps:
1. Tune vs enemy 4
2. Train 10 times vs 3 enemies
3. Per enemy 10 line max, 10 lines mean
4. Take average of all max and mean values per enemy.
5. Take std of max and mean values per enemy.
6. Repeat for EA2 and combine final plot in final plot
7. Repeat for all enemies.
8. Take best individual out of all 10 sims and play against enemy 5 times and measure the gain.
Gain = Player Lifepoints - Enemy Lifepoints
9. Generate barplot
10. Repeat for all enemies.
'''

__enemies = [4, 5, 7]

SA_agent = GASpecialistSA(SA=True, experiment_name="test_run_group55_SA") # SA agent
Normal_agent = GASpecialistSA(SA=False, experiment_name="test_run_group55") # Regular agent

def train_agent(agent: GASpecialistSA,
                enemies: list[int],
                overwrite: bool = False,
                max_runs: int = 10):

    current_log = []

    # If there is no train log, create one.
    if not os.path.exists(agent.experiment_name):
        os.makedirs(agent.experiment_name)

        with open(agent.experiment_name + "/trainlog.txt", "a") as file_aux:
            file_aux.write(f'ENEMY RUNS\n')
            # Create spots for every possible enemy.
            for i in range(1,9):
                file_aux.write(f'{i}: 0\n')

    # Load the train log
    with open(agent.experiment_name + "/trainlog.txt", "r") as file_aux:
        temp_log = file_aux.readlines()
        current_log = list(map(str.split, temp_log[1:]))

    for enemy in enemies:

        if not os.path.exists(agent.experiment_name + f"/enemy_{enemy}"):
            os.makedirs(agent.experiment_name + f"/enemy_{enemy}")

        passed_runs = int(current_log[enemy - 1][1])
        allowed_runs = max_runs # How many runs should the agent have left vs an enemy.

        if (passed_runs == max_runs) & (overwrite == False):
            # If the agent already has ran 10 times vs an enemy skip it. Unless overwrite
            print(f"Agent has already ran {max_runs} runs vs {enemy}")
            continue
        elif not overwrite:
            # If the agent has previous runs vs this enemy, run n times till you hit the max.
            allowed_runs = max_runs - passed_runs
            print(f"Agent has {allowed_runs} runs vs {enemy} remaining")

        for i in range(passed_runs, max_runs):
            agent.run_experiment(enemy=enemy, mode="Train", run=i)

            with open(agent.experiment_name + "/trainlog.txt", "r+") as file_aux:
                write_log = file_aux.readlines()
                file_aux.seek(0)
                write_log[enemy] = f"{enemy}: {i+1}\n"

                file_aux.writelines(write_log)

                file_aux.truncate()


# TODO: Create Test function (Test agent)

# Finds the highest fitness out of all best candidates per enemy
def find_highest_fitness(fitness_scores: list[str]) -> int:
    fitness_list = [x.split()[1] for x in fitness_scores]
    best_fit = max([float(x) for x in fitness_list])
    return fitness_list.index(str(best_fit))

# Start training phase
def test_best_agent(agent: GASpecialistSA, enemies: list[int]):

    # Create new result file
    with open(f"{agent.experiment_name}/testing_results.txt", "w") as file_aux:
        file_aux.write("{:<10} {:<10} {:<10} {:<10} {:<10}\n".format(
            "FITNESS", "PLAYER_HP", "ENEMY_HP", "TIME", "ENEMY"
        ))

    # Do 5 runs per enemy
    for e in enemies:

        best_ind_index = 0 # default value
        results = []

        # Get the best individual to load in for testing.
        with open(f"{agent.experiment_name}/enemy_{e}/best_fitness.txt", "r") as file_aux:
            best_fitness = file_aux.readlines()
            best_ind_index = find_highest_fitness(best_fitness)

        for i in range(0, 5):
            result = agent.run_experiment(mode="Test", enemy=e, run=i, best_ind_idx=best_ind_index)
            results.append(result)

        with open(f"{agent.experiment_name}/testing_results.txt", "a") as file_aux:
            for t in results:
                file_aux.write(f'{t[0]:<10.6f} {t[1]:<10.6f} {t[2]:<10} {t[3]:<10} {e:<10}\n')

# TODO: Create plots

def plot_training_results():
    pass

for ga_agent in [SA_agent, Normal_agent]:
    train_agent(ga_agent, __enemies)
    test_best_agent(ga_agent, __enemies)