import math
import random
import time
from DataTypes import distributions
import numpy as np

from Selection_methods import selection_methods
from settings import GA_MAXITER

import numpy

"""
        This function greets a person with their name and age.

        @param n: number of parameters.
        @param inputType: type of input data.
        @param max_iter: number of iterations
        @param number_of_clustures :number of clusters to be associated with the data points 
        
        @return: A string greeting the person.
"""
class algortithem:

    def __init__(self, n, inputType, max_iter, number_of_clustures,input=None):
        self.inputParameters = list(range(n))
        self.buffer = list(range(n))
        self.fitness_array = numpy.zeros(n)

        self.n = n

        self.iteration = 0  # current iteration that went through the algorithm
        self.inputType = inputType

        self.tick = 0
        self.sol_time = 0
        self.max_iter = max_iter
        self.solution = inputType()
        self.output = []
        self.output2 = []
        self.iter = []
        self.solution2 = self.inputType()

        # number of clusters
        self.k = number_of_clustures

        self.distributions = []

        self.n_inputs = [] if input==None else input
        self.initInput()

    # create random sigma and miu values for first iteration!
    def createSigmasAndMius(self):
        self.distributions = [distributions() for i in range(self.k)]

    def sorting(self, population):
        # todo if you want to use it ,create a <= operator in input type
        return sorted(population, reverse=False)

    def initInput(self):
        self.n_inputs = [self.inputType() for i in range(self.k)]

    def initAllAlgoParameters(self):
        self.createSigmasAndMius()
        self.initInput() if self.n_inputs ==[] else None

    def eStep(self):
        for input in self.n_inputs:
            for distribution in self.distributions:
                pass
            #todo do the calculations and then do the m step calculations.


    def mstep(self):
        # todo implement the m step of the algorithm
        pass
    def calculate_Pie(self,input):


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
