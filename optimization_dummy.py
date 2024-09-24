###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Group 55       			                                          #
# TODO: Add email  				                                              #
###############################################################################
import array
# imports framework
import sys
from random import random, uniform
from venv import create

import numpy

from evoman.environment import Environment
from demo_controller import player_controller
from deap import algorithms, base, benchmarks, creator, tools

# imports other libs
import numpy as np
import os

experiment_name = 'optimization_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

# runs simulation
def simulation(x):
    f,p,e,t = env.play(pcont=x)
    return f,

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def mutate():
    pass

def crossover():
    pass

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", uniform, -1, 1)

def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=n_vars)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    toolbox.register("evaluate", simulation)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # start writing your own code from here

    pop = toolbox.population(n=100)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50,
                                   stats=stats, verbose=True)

    print("FINISHED!")
    print("##############################")

if __name__ == '__main__':
    main()