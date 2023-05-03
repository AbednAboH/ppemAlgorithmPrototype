# class for integer objects as input for the algorithm
import random

import numpy as np

from settings import MAXSIGMA, MAXMU


# todo change this in the future
class distributions:
    def __init__(self,data,numberOfClusters):
        self.sigma = np.random.uniform(low=min(data), high=max(data), size=numberOfClusters)
        self.mu = np.random.uniform(low=0, high=1, size=numberOfClusters)
    def normalDistribution(self,data,numberOfClusters):
        self.sigma = np.random.uniform(low=min(data), high=max(data), size=numberOfClusters)
        self.mu = np.random.uniform(low=0, high=1, size=numberOfClusters)

# weights /initial pi value
class PiCreation:
    def normalDistribution(self, numberOfClusters):
        return np.ones(numberOfClusters) / numberOfClusters

class oneDimentional:
    

# one dimensional input:
class pointDataType:

    def __init__(self, numberOfClusters):
        # create the input for the algorithm ,can be change in the future

        self.object = random.gauss(mu=MAXMU, sigma=MAXSIGMA)
        # Working with kind of Random Starts start paradigm

        # Array of pi-s
        self.propabilies = PiCreation().normalDistribution(numberOfClusters)

    def create_object(self, target=None, options=None):
        #  if we want to change how we create an object in the future
        self.object = target

    def createNewPropabilies(self, prob):
        self.propabilies = prob


# class for a vector of input, uses target as an object
class vector_dataType(pointDataType):
    def __init__(self):
        super(vector_dataType, self).__init__()
        self.object = []

    def create_object(self, target=None, options=None):
        self.object = target

    def setDistributionAssociation(self, distribution):
        pass
