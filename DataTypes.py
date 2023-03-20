
# class for integer objects as input for the algorithm
class dataType:

    def __init__(self):
        self.object=None

    def create_object(self, target_size, target, options=None):
        return self.object
    def calculate_sigma(self):
        pass
# class for a vector of input, uses target as an object
class vector_dataType(dataType):
    def __init__(self):
        super(vector_dataType, self).__init__()
        self.object=[]
    def create_object(self, target_size, target, options=None):
        self.object=target
    def calculate_sigma(self):
        pass

