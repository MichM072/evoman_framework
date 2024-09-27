from sklearn.model_selection import ParameterGrid
import numpy as np
import os
import json  # For converting dictionaries to string format
from GA_Spec_SA import GASpecialistSA


class Tuner:
    def __init__(self, agent: GASpecialistSA, param_grid: dict, enemies: list):
        self.agent = agent
        self.param_grid = param_grid
        self.enemies = enemies
        self.results = []

    def evaluate_params(self, max_mutpb, min_mutpb, cooling_rate, enemy, run):
        self.agent.max_mutpb = max_mutpb
        self.agent.min_mutpb = min_mutpb
        self.agent.cooling_rate = cooling_rate

        self.agent.run_experiment(enemy, mode="Tune", run=run, best_ind_idx=0)

        # Saves the best fitness
        file_path = self.agent.experiment_name + f"/tune_best_{enemy}.txt"

        best_fitness = np.loadtxt(file_path)[0]

        return best_fitness

    def tune_parameters(self):
        """Perform grid search across all combinations of parameters"""
        for enemy in self.enemies:
            print(f"\nTuning for enemy {enemy}")
            i = 0
            for params in ParameterGrid(self.param_grid):
                print(f"Evaluating params: {params}")

                # Evaluate each combination of parameters
                fitness = self.evaluate_params(
                    params["max_mutpb"],
                    params["min_mutpb"],
                    params["cooling_rate"],
                    enemy,
                    run = i
                )

                result = {
                    "enemy": enemy,
                    "max_mutpb": params["max_mutpb"],
                    "min_mutpb": params["min_mutpb"],
                    "cooling_rate": params["cooling_rate"],
                    "fitness": fitness,
                }
                self.results.append(result)
                i += 1

        # Saving
        results_str = [json.dumps(result) for result in self.results]

        np.savetxt(
            self.agent.experiment_name + "/tuning_results.txt",
            results_str,
            fmt="%s",
        )

        print("\nParameter tuning complete. Results saved.")

    def get_results(self):
        return self.results
