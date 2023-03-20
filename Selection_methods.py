from math import sqrt
import numpy
from random import randint, random, sample
from settings import RAND, SUS, RWS, TOUR


class selection_methods:
    # static propabilies list

    ranks = []

    def __init__(self):
        self.method = {RAND: self.random_selection, SUS: self.SUS, RWS: self.RWS,
                       TOUR: self.tournement,"tour2":self.tournement2}

    def random_selection(self, population, fitness_array, k=10):
        # select random places from half the population
        popsize = len(population)
        i1 = randint(0, int(popsize *0.1))
        i2 = randint(0,  int(popsize *0.1))
        return population[i1], population[i2]

    def SUS(self, population, fitness_array, k=10):
        range_of_choices = len(fitness_array)
        chosen = numpy.random.choice(range_of_choices, p=fitness_array)
        chosen2 = (chosen + len(fitness_array) // 2) % len(fitness_array)
        return population[chosen], population[chosen2]

    def RWS(self, population, fitness_array, k=10):
        # check the +1 !
        range_of_choices = len(fitness_array)
        # roll the rullette
        chosen = numpy.random.choice(range_of_choices, p=fitness_array)
        chosen2 = numpy.random.choice(range_of_choices, p=fitness_array)
        return population[chosen], population[chosen2]

    def tournement(self, population, fitness_array, k=15):
        # get samples from population
        k1=min(k,len(population))
        participants1 = sample(population, k1)
        participants2 = sample(population, k1)
        # return minumum from 2 samples
        return min(participants1), min(participants2)
    def tournement2(self, population, fitness_array, k=100):
        sets=[]
        for i in range(k):
            p1,p2=self.tournement(population, fitness_array, k=15)
            sets.append(p1)
            sets.append(p2)
        return sets
    def spin_the_rulette(self, population, mean):
        # spin the wheel:
        fitness_array = numpy.array([1 / linear_scale((citizen.fitness, 0.5, 1)) for citizen in population])
        fitness_sum = fitness_array.sum()
        self.ranks = [1 / linear_scale((citizen.fitness, 0.5, 1)) / fitness_sum for citizen in population]


def linear_scale(x):
    return x[0] * x[1] + x[2]


# linear_scale = lambda x:x[0]*x[1]+x[2]
e_scale = lambda x: sqrt(x)
