###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# Test : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Group 55   			                                              #			                                  #
###############################################################################

from GA_Spec_SA import GASpecialistSA
import matplotlib

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

enemies = [4, 5, 7]

SA_agent = GASpecialistSA(SA=True)
Normal_agent = GASpecialistSA(SA=False)

for e in enemies:
    Normal_agent.run_experiment(e)

def train_agent():
    pass
