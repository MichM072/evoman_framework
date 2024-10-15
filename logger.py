import numpy as np
from deap.tools import HallOfFame

class Logger:
    def __init__(
            self,
            experiment_name: str,
            mode: str,
            enemy: int
    ):
        self.experiment_name = experiment_name
        self.mode = mode.lower()
        self.enemy = enemy

    def log_initial_population_fitness(self) -> None:
        with open(self._get_log_filename(), "a") as file:
            file.write("\n{:<10} {:<10} {:<10} {:<10} ENEMY: {}\n".format(
                "GENERATION", "BEST", "MEAN", "STD", self.enemy))

    def log_generation_statistics(
            self,
            generation: np.array,
            best: np.array,
            record: np.array
    ) -> None:
        best_fitness = best[0].fitness.values[0]

        print(f'\nGENERATION {generation} best: {round(best_fitness, 6)} avg: {round(record["avg"], 6)} std: {round(record["std"], 6)} enemy: {self.enemy}')

        with open(self._get_log_filename(), "a") as file:
            file.write(
                f'\n{generation:<10} {best_fitness:<10.6f} {record["avg"]:<10.6f} {record["std"]:<10.6f}'
            )

    def save_results(
            self,
            best: HallOfFame,
            run: int,
    ) -> None:
        np.savetxt(f"{self._get_enemy_directory()}/best_{run}.txt", best[0])

        with open(f"{self._get_enemy_directory()}/best_fitness.txt", "a") as file:
            file.write(f"{run}: {best[0].fitness.values[0]}\n")

        print(f"\nBest fitness achieved: {best[0].fitness.values[0]}")


    def _get_log_filename(self) -> str:
        return f"{self.experiment_name}/{self.mode}_results.txt"

    def _get_enemy_directory(self) -> str:
        return f"{self.experiment_name}/enemy_{self.enemy}"