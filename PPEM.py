from Partial_EM import Partial_EM
import numpy as np


class Partial_PPEM(Partial_EM):
    def __init__(self, n, inputDimentions: int = 2, max_iter: int = 1, number_of_clustures: int = 2, eps: float = 1e-5,
                 epsilonExceleration: bool = True,
                 input: np.array = None, plottingTools: bool = False, plot_name="", encrypt=None):
        """
        Parameters:
            n:number of parameters.
            max_iter:number of iterations
            number_of_clustures :number of clusters to be associated with the data points
            input :input of the PPEM algorithm1 when we use server client model
            inputDimentions:the dimensions of the input array
            epsilonExceleration:a boolean that enables/disables convergence with epsilons aid
            eps:the value of epsilon that helps with the convergence criteria
            plottingTools:if True the algorithm plots a gif of the EM process of the algorithm

          """
        super(Partial_PPEM, self).__init__(n=n, inputDimentions=inputDimentions, max_iter=max_iter, number_of_clustures=number_of_clustures, eps=eps, epsilonExceleration= epsilonExceleration,
                                           input=input, plottingTools=plottingTools, plot_name=plot_name)
        # encryption unit for encrypting the data for each client
        self.encryption_unit = encrypt
        self.qisaEncrypted = None
    
    def update_encryption(self, context):
        self.encryption_unit = context

    def mStep_epsilon(self):
        """ calculate the sum of the responsibilities ,and the uppder side of the means equation meaning q_i,s,a* X_i then return them to the server"""
        a,b,c = super(Partial_PPEM, self).mStep_epsilon()
        return self.encryption_unit.CKKS_encrypt(a), self.encryption_unit.CKKS_encrypt(b),self.encryption_unit.CKKS_encrypt(c)
     
    def update(self, a_all,b_all,c_all,n):
        a=self.encryption_unit.decrypt(a_all)
        b=self.encryption_unit.decrypt(b_all)

        c=self.encryption_unit.decrypt(c_all)
        n_=self.encryption_unit.decrypt(n)
        super(Partial_PPEM, self).update(a,b,c,n_)
