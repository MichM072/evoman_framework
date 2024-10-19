import numpy as np
import os
import csv
from evoman.environment import Environment
from demo_controller import player_controller

os.environ["SDL_VIDEODRIVER"] = "dummy"


def test_best_solution_against_all_enemies(best_solution_path, experiment_name, enemies=range(1, 9)):
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    best_solution = np.loadtxt(best_solution_path)

    def create_env_for_enemy(enemy_number):
        return Environment(
            experiment_name=experiment_name,
            enemies=[enemy_number],
            multiplemode="no",
            playermode="ai",
            player_controller=player_controller(10),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False,
            randomini='no'
        )

    results = []

    for enemy in enemies:
        env = create_env_for_enemy(enemy)
        fitness, player_life, enemy_life, time = env.play(best_solution)
        results.append([enemy, player_life, enemy_life])

    return results


def save_results_to_csv(results, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['Enemy', 'Player health', 'Enemy health'])

        for result in results:
            writer.writerow(result)


best_solution_path = 'best/best.txt'
experiment_name = 'best'

results = test_best_solution_against_all_enemies(best_solution_path, experiment_name)
save_results_to_csv(results, 'best/results.csv')
