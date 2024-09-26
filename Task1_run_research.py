###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# Test : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Group 55   			                                              #			                                  #
###############################################################################

from GA_Spec_SA import GASpecialistSA

# Test implementation

enemies = [4, 5, 7]

SA_agent = GASpecialistSA(SA=True)
Normal_agent = GASpecialistSA(SA=False)

for e in enemies:
    Normal_agent.run_experiment(e)
