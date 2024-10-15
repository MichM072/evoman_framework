###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# Test : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Group 55   			                                              #			                                  #
###############################################################################
import numpy as np

from GA_Spec_SA import GASpecialistSA
import matplotlib
from Tuning_SA import Tuner
import argparse
import os
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

"""
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
"""

parser = argparse.ArgumentParser(description="Run parameter tuning or experiments")
parser.add_argument(
    "--tune",
    type=bool,
    default=False,
    help="Set this to True for running parameter tuning for the Simulated Annealing, False for just running the experiment",
)
args = parser.parse_args()
tuning_sa = args.tune

__enemies = [2, 4, 8]

# SA_agent = GASpecialistSA(sa=True)
# Normal_agent = GASpecialistSA(sa=False)


param_grid = {
    "max_mutpb": [0.7, 0.9],  # 0.3, 0.5,
    "min_mutpb": [0.01, 0.05],  # , 0.07, 0.1
    "cooling_rate": [0.97, 0.99],  # 0.93, 0.95,
}


# __enemies = [4, 5, 7]

SA_agent = GASpecialistSA(sa=True, experiment_name="test_run_group55_SA")  # SA agent
Normal_agent = GASpecialistSA(
    sa=False, experiment_name="test_run_group55"
)  # Regular agent


def train_agent(
    agent: GASpecialistSA,
    enemies: list[int],
    overwrite: bool = False,
    max_runs: int = 10,
):
    if os.path.exists(f"{agent.experiment_name}/train_results.txt"):
        os.remove(f"{agent.experiment_name}/train_results.txt")

    current_log = []

    # If there is no train log, create one.
    if not os.path.exists(agent.experiment_name):
        os.makedirs(agent.experiment_name)

        with open(agent.experiment_name + "/trainlog.txt", "a") as file_aux:
            file_aux.write(f"ENEMY RUNS\n")
            # Create spots for every possible enemy.
            for i in range(1, 9):
                file_aux.write(f"{i}: 0\n")

    # Load the train log
    with open(agent.experiment_name + "/trainlog.txt", "r") as file_aux:
        temp_log = file_aux.readlines()
        current_log = list(map(str.split, temp_log[1:]))

    for enemy in enemies:

        if not os.path.exists(agent.experiment_name + f"/enemy_{enemy}"):
            os.makedirs(agent.experiment_name + f"/enemy_{enemy}")

        passed_runs = int(current_log[enemy - 1][1])
        allowed_runs = max_runs  # How many runs should the agent have left vs an enemy.

        if (passed_runs == max_runs) & (overwrite == False):
            # If the agent already has ran 10 times vs an enemy skip it. Unless overwrite
            print(f"Agent has already ran {max_runs} runs vs {enemy}")
            continue
        elif not overwrite:
            # If the agent has previous runs vs this enemy, run n times till you hit the max.
            allowed_runs = max_runs - passed_runs
            print(f"Agent has {allowed_runs} runs vs {enemy} remaining")

        with mp.Pool(mp.cpu_count()) as pool:
            results = [
                pool.apply_async(agent.run_experiment, (enemy, "Train", i, 0))
                for i in range(passed_runs, max_runs)
            ]

            # Collect the results
            for i, result in enumerate(results):
                result.get()  # Ensure each experiment finishes

                # Update log
                with open(agent.experiment_name + "/trainlog.txt", "r+") as file_aux:
                    write_log = file_aux.readlines()
                    file_aux.seek(0)
                    write_log[enemy] = f"{enemy}: {passed_runs + i + 1}\n"
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
    with open(f"./{agent.experiment_name}/testing_results.txt", "w") as file_aux:
        file_aux.write("FITNESS,PLAYER_HP,ENEMY_HP,TIME,ENEMY\n")

    # Do 5 runs per enemy
    for e in enemies:

        best_ind_index = 0  # default value
        results = []

        # Get the best individual to load in for testing.
        with open(
            f"{agent.experiment_name}/enemy_{e}/train_best_fitness.txt", "r"
        ) as file_aux:
            best_fitness = file_aux.readlines()
            print(best_fitness)
            best_ind_index = find_highest_fitness(best_fitness)

        for i in range(0, 5):
            result = agent.run_experiment(
                mode="Test", enemy=e, run=i, best_ind_idx=best_ind_index
            )
            print(result)
            results.append(result)

        with open(f"./{agent.experiment_name}/testing_results.txt", "a") as file_aux:
            for t in results:
                file_aux.write(f"{t[0]:.6f},{t[1]:.6f},{t[2]},{t[3]},{e}\n")


# TODO: Create plots
def get_fitness(agent, __enemies, average=False):

    enemy_1_fitness = []  # enemy 4
    enemy_2_fitness = []  # enemy 6
    enemy_3_fitness = []  # enemy 8
    with open(f"./{agent.experiment_name}/testing_results.txt", "r") as file:
        results_file = file.readlines()

    for i in results_file[1:]:
        i = i.split(",")
        fitness = float(i[0])
        enemy = int(i[4])
        if enemy == __enemies[0]:
            enemy_1_fitness.append(fitness)
        elif enemy == __enemies[1]:
            enemy_2_fitness.append(fitness)
        elif enemy == __enemies[2]:
            enemy_3_fitness.append(fitness)

    # Calculate average
    avg_fitness_1 = sum(enemy_1_fitness) / 5
    avg_fitness_2 = sum(enemy_2_fitness) / 5
    avg_fitness_3 = sum(enemy_3_fitness) / 5

    avg_ = [avg_fitness_1, avg_fitness_2, avg_fitness_3]

    fitness_ = [enemy_1_fitness, enemy_2_fitness, enemy_3_fitness]

    if average == True:
        return avg_
    else:
        return fitness_


def get_time(agent, __enemies, average=False):

    enemy_1_time = []  # enemy 4
    enemy_2_time = []  # enemy 6
    enemy_3_time = []  # enemy 8
    with open(f"./{agent.experiment_name}/testing_results.txt", "r") as file:
        results_file = file.readlines()

    for i in results_file[1:]:
        i = i.split(",")
        time = float(i[3])
        enemy = int(i[4])
        if enemy == __enemies[0]:
            enemy_1_time.append(time)
        elif enemy == __enemies[1]:
            enemy_2_time.append(time)
        elif enemy == __enemies[2]:
            enemy_3_time.append(time)

    # Calculate average
    avg_time_1 = sum(enemy_1_time) / 5
    avg_time_2 = sum(enemy_2_time) / 5
    avg_time_3 = sum(enemy_3_time) / 5

    avg_ = [avg_time_1, avg_time_2, avg_time_3]

    time_ = [enemy_1_time, enemy_2_time, enemy_3_time]

    if average == True:
        return avg_
    else:
        return time_


def get_gain(agent, enemies, average=False):

    enemy_1_gain = []  # enemy 4
    enemy_2_gain = []  # enemy 6
    enemy_3_gain = []  # enemy 8
    with open(f"./{agent.experiment_name}/testing_results.txt", "r") as file:
        results_file = file.readlines()

    for i in results_file[1:]:
        i = i.split(",")
        enemy = int(i[4])
        health = float(i[1])
        enemy_health = float(i[2])

        gain = health - enemy_health

        if enemy == enemies[0]:
            enemy_1_gain.append(gain)
        elif enemy == enemies[1]:
            enemy_2_gain.append(gain)
        elif enemy == enemies[2]:
            enemy_3_gain.append(gain)

    avg_gain_1 = sum(enemy_1_gain) / 5
    avg_gain_2 = sum(enemy_1_gain) / 5
    avg_gain_3 = sum(enemy_1_gain) / 5

    avg_ = [avg_gain_1, avg_gain_2, avg_gain_3]
    gains = [enemy_1_gain, enemy_2_gain, enemy_3_gain]

    if average == True:
        return avg_
    else:
        return gains


def train_to_csv(agent):
    with open(f"./{agent.experiment_name}/training_results.txt", "r") as infile:
        lines = infile.readlines()

    with open(f"./{agent.experiment_name}/training_for_processing.csv", "w") as f:
        # Write the header
        f.write(",".join(lines[0].split()) + "\n")

        # Process the rest
        for line in lines[1:]:
            if any(char.isdigit() for char in line):
                f.write(",".join(line.split()) + "\n")


def get_train_vals(agent):
    train_to_csv(agent)

    with open(f"./{agent.experiment_name}/training_for_processing.csv", "r") as file:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header

        for row in reader:
            run = int(row[0])
            generation = int(row[1])
            best = float(row[2])
            enemy = int(row[4])

            # Append the generation and best score to the corresponding run and enemy
            data[run][enemy].append((generation, best))

    # Sort the generations in ascending order for each run and enemy
    for run in data:
        for enemy in data[run]:
            data[run][enemy].sort(key=lambda x: x[0])  # Sort by generation (x[0])

    # return data


def plot_avg_bar(
    function_,
    xlabel,
    ylabel,
    title,
    __enemies=__enemies,
    labels=["Enemy 4", "Enemy 6", "Enemy 8"],
):
    SAavg_ = function_(SA_agent, __enemies, average=True)
    avg_ = function_(Normal_agent, __enemies, average=True)

    labels = labels

    x = np.arange(len(labels))
    fig, ax = plt.subplots()

    ax.grid(True, axis="y", linestyle="--", zorder=0)

    width = 0.30
    bars_SA = ax.bar(
        x - width / 2,
        SAavg_,
        width,
        label="Simulated Annealing",
        color="gray",
        zorder=2,
    )
    bars_normal = ax.bar(
        x + width / 2,
        avg_,
        width,
        label="Standard",
        color="lightgray",
        zorder=2,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()  # Leave extra space at the top for the title

    directory = "graphs_images"
    if not os.path.exists(directory):
        os.makedirs(directory)

    plot_path = os.path.join(directory, f"{title}.png")
    plt.savefig(plot_path, format="png")
    plt.close(fig)


def plot_boxplot(
    function_,
    xlabel,
    ylabel,
    title,
    __enemies=__enemies,
    tick_labels=["4", "6", "8"],
):
    SAavg_ = function_(SA_agent, __enemies)
    avg_ = function_(Normal_agent, __enemies)

    data = []
    for i in range(len(__enemies)):  # stores the data in apirs per enemy
        data.append(SAavg_[i])
        data.append(avg_[i])

    boxplot_labels = []  # Makes the labels appear in pairs
    for label in tick_labels:
        boxplot_labels.extend([f"{label}", f"{label}"])

    median_color = "blue"
    mean_color = "green"
    fig, ax = plt.subplots()
    boxplots = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=boxplot_labels,
        showmeans=True,
        meanline=True,
        meanprops=dict(color=mean_color, linewidth=1.5),
        medianprops=dict(color=median_color),
    )

    color = ["gray", "lightgray"]
    for i, box in enumerate(boxplots["boxes"]):
        if i % 2 == 0:  # if SA
            box.set_facecolor(color[0])
        else:  # if normal
            box.set_facecolor(color[1])

    mean_handle = plt.Line2D(
        [0], [0], color=mean_color, linestyle="--", lw=1.5, label="Mean"
    )
    median_handle = plt.Line2D([0], [0], color=median_color, lw=1.5, label="Median")

    mean_handle = plt.Line2D(
        [0], [0], color=mean_color, linestyle="--", lw=1.5, label="Mean"
    )
    median_handle = plt.Line2D([0], [0], color=median_color, lw=1.5, label="Median")
    sa_handle = plt.Line2D([0], [0], color="gray", lw=4, label="SA")
    normal_handle = plt.Line2D([0], [0], color="lightgray", lw=4, label="Normal")

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, axis="y", linestyle="--")
    ax.legend(
        handles=[sa_handle, normal_handle, mean_handle, median_handle],
        loc="lower right",
    )

    directory = "graphs_images"
    if not os.path.exists(directory):
        os.makedirs(directory)

    plot_path = os.path.join(directory, f"{title}.png")
    plt.savefig(plot_path, format="png")
    plt.close(fig)


def plot_boxplot_global(
    function_,
    xlabel,
    ylabel,
    title,
    __enemies=__enemies,
    tick_labels=["SA", "Normal"],
):
    SAavg_ = function_(SA_agent, __enemies)
    avg_ = function_(Normal_agent, __enemies)
    SAavg_flat = [item for sublist in SAavg_ for item in sublist]
    avg_flat = [item for sublist in avg_ for item in sublist]

    fig, ax = plt.subplots(figsize=(8, 6))

    boxplots = ax.boxplot(
        [SAavg_flat, avg_flat],
        patch_artist=True,
        tick_labels=tick_labels,
        boxprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2),
        medianprops=dict(linewidth=2),
        widths=0.3,
    )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, axis="y", linestyle="--")

    for patch, color in zip(boxplots["boxes"], ["gray", "lightgray"]):
        patch.set_facecolor(color)

    plt.tight_layout()

    directory = "graphs_images"
    if not os.path.exists(directory):
        os.makedirs(directory)

    plot_path = os.path.join(directory, f"{title}_global.png")
    plt.savefig(plot_path, format="png")
    plt.close(fig)


def plot_line_chart_per_enemy(
    function_, xlabel, ylabel, title, __enemies=__enemies, tick_labels=["4", "6", "8"]
):
    SAavg_ = function_(SA_agent, __enemies)
    avg_ = function_(Normal_agent, __enemies)

    # Loops over each enemy and creates a plot
    for i, enemy in enumerate(__enemies):
        fig, ax = plt.subplots()

        # Plot data for SA
        ax.plot(
            range(1, len(SAavg_[i]) + 1),
            SAavg_[i],
            label="SA",
            color="gray",
            marker="o",
        )

        # Plot data for Normal agent
        ax.plot(
            range(1, len(avg_[i]) + 1),
            avg_[i],
            label="Normal",
            color="lightgray",
            marker="o",
            linestyle="--",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} - Enemy {tick_labels[i]}")
        ax.set_xticks(range(1, len(SAavg_[i]) + 1))
        ax.legend()

        plt.savefig(f"{title} Enemy {tick_labels[i]}.png", format="png")


if __name__ == "__main__":

    if tuning_sa:
        tuner = Tuner(SA_agent, param_grid, __enemies)

        tuner.tune_parameters()
    else:
        for ga_agent in [SA_agent, Normal_agent]:
            # train_agent(ga_agent, __enemies)
            # test_best_agent(ga_agent, __enemies)


plot_avg_bar(
    function_=get_fitness,
    xlabel="Enemies",
    ylabel="Average Fitness",
    title="Fitness per Enemy and per Algorithm",
    __enemies=__enemies,
    labels=["Enemy 4", "Enemy 6", "Enemy 8"],
)

plot_avg_bar(
    function_=get_gain,
    xlabel="Enemies",
    ylabel="Average Gain",
    title="Gain per Enemy and per Algorithm ",
    __enemies=__enemies,
    labels=["Enemy 4", "Enemy 6", "Enemy 8"],
)

plot_boxplot(
    function_=get_gain,
    xlabel="Enemies",
    ylabel="Gains",
    title="Gain per Enemy and per Algorithm",
    __enemies=__enemies,
)

plot_avg_bar(
    function_=get_time,
    xlabel="Enemies",
    ylabel="Average Time",
    title="Time per Enemy and per Algorithm",
    __enemies=__enemies,
)

plot_boxplot(
    function_=get_time,
    xlabel="Enemies",
    ylabel="Gains",
    title="Time per Enemy and per Algorithm",
    __enemies=__enemies,
)

plot_boxplot_global(
    function_=get_gain,
    xlabel="Algorithm",
    ylabel="Gains",
    title="Gain per Algorithm",
    __enemies=__enemies,
)

plot_boxplot_global(
    function_=get_time,
    xlabel="Algorithm",
    ylabel="Gains",
    title="Time per Algorithm",
    __enemies=__enemies,
)


plot_line_chart_per_enemy(
    function_=get_gain,
    xlabel="Runs",
    ylabel="Average Gain",
    title="Line Gain between the SA and the Normal Agents",
    __enemies=__enemies,
    tick_labels=["Enemy 4", "Enemy 6", "Enemy 8"],
)

plot_line_chart_per_enemy(
    function_=get_time,
    xlabel="Runs",
    ylabel="Average Gain",
    title="Line Time between the SA and the Normal Agents",
    __enemies=__enemies,
    tick_labels=["Enemy 4", "Enemy 6", "Enemy 8"],
)
