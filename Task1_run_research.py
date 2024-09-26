###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# Test : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Group 55   			                                              #			                                  #
###############################################################################

from GA_Spec_SA import GASpecialistSA

# Test implementation

enemies = [1, 4, 6]

SA_agent = GASpecialistSA()

for e in enemies:
    SA_agent.run_experiment(e)