# class for integer objects as input for the algorithm
import random


class pointDataType:

    def __init__(self):
        self.object = None
        self.associatedDistribution=None
    def create_object(self, target=None, options=None):
        self.object=target
    def setDistributionAssociation(self,distribution):
        pass


# class for a vector of input, uses target as an object
class vector_dataType(pointDataType):
    def __init__(self):
        super(vector_dataType, self).__init__()
        self.object = []

    def create_object(self, target=None, options=None):
        self.object = target

    def setDistributionAssociation(self,distribution):
        pass
