import math
import random
import time
from DataTypes import distributions
import numpy as np

from Selection_methods import selection_methods
from settings import GA_MAXITER

import numpy


class algortithem:
    def __init__(self, target, tar_size, n, inputType, selection, max_iter, number_of_clustures,
                 cluster_propabilities):
        self.inputParameters = list(range(n))
        self.buffer = list(range(n))
        self.fitness_array = numpy.zeros((n))
        self.target = target
        self.target_size = tar_size
        self.n = n

        self.iteration = 0  # current iteration that went through the algorithm
        self.inputType = inputType

        self.selection_methods = selection_methods()
        self.selection = selection

        self.tick = 0
        self.sol_time = 0
        self.max_iter = max_iter
        self.solution = inputType()
        self.output = []
        self.output2 = []
        self.iter = []
        self.solution2 = self.inputType()

        # Pie in the equations
        self.cluster_porbabilities = cluster_propabilities

        # number of clustures
        self.k = number_of_clustures

        self.distributions = self.createSigmasAndMius()

    # create random sigma and miu values for first iteration!
    def createSigmasAndMius(self):
        return [distributions() for i in range(self.k)]

    def sorting(self, population):
        # todo if you want to use it ,create a <= operator in input type
        return sorted(population, reverse=False)

    def initInput(self):
        for new_input in range(self.n):


    def eStep(self):
        # todo implement the e step on specific
        pass

    def mstep(self):
        # todo implement the m step of the algorithm
        pass

    def handle_initial_time(self):
        self.tick = time.time()
        self.sol_time = time.perf_counter()

    def handle_prints_time(self):
        runtime = time.perf_counter() - self.sol_time
        clockticks = time.time() - self.tick
        print_B(self.solution)
        # print_mean_var((self.pop_mean, variance((self.pop_mean, self.solution.fitness))))
        print_time((runtime, clockticks))

    def algo(self, i):
        self.eStep()
        self.mstep()

    def stopage(self, i):
        return False

    def solve(self):
        self.handle_initial_time()
        self.init_inputAndCreateSigmas()
        for i in range(self.max_iter):

            self.iteration += 1
            self.algo(i)
            # self.output.append(self.solution.fitness)

            self.iter.append(i)
            self.handle_prints_time()
            if self.stopage(i) or i == self.max_iter - 1:
                print(" number of generations : ", i)
                self.handle_prints_time()
                break

        return self.output, self.iter, self.solution, self.output2, self.solution2, self.inputParameters


# print_B = lambda x: print(f" Best:{len(x.object)} ,fittness: {x.fitness} ", end=" ")
print_B = lambda x: print(f" Best:{x} ,\nfittness: {x.fitness} ", end=" ")
# print_B = lambda x: print(f" Best: {x.object} ,fittness: {x.fitness} ", end=" ")

#  prints mean and variance
print_mean_var = lambda x: print(f"Mean: {x[0]} ,Variance: {x[1]}", end=" ")
# prints time
print_time = lambda x: print(f"Time :  {x[0]}  ticks: {x[1]}")
# calculates variance
variance = lambda x: math.sqrt((x[0] - x[1]) ** 2)
