###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# Test : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Group 55   			                                              #			                                  #
###############################################################################

from GA_Spec_SA import GASpecialistSA
import matplotlib
from Tuning_SA import Tuner
import argparse

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

enemies = [4, 6, 8]

SA_agent = GASpecialistSA(sa=True)
Normal_agent = GASpecialistSA(sa=False)


param_grid = {
    "max_mutpb": [0.3, 0.5, 0.7, 0.9],
    "min_mutpb": [0.01, 0.05, 0.07, 0.1],
    "cooling_rate": [0.93, 0.95, 0.97, 0.99],
}

if tuning_sa:
    tuner = Tuner(SA_agent, param_grid, enemies)

    tuner.tune_parameters()
else:
    for e in enemies:
        SA_agent.run_experiment(e)
