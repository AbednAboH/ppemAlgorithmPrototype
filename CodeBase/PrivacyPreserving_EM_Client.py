from CodeBase.Federated_EM_Client import Partial_EM
import numpy as np
import copy
import concurrent.futures
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
        self.encryption_unit = copy.deepcopy(context)

    def mStep_epsilon(self):
        """ calculate the sum of the responsibilities ,and the uppder side of the means equation meaning q_i,s,a* X_i then return them to the server"""
        a,b,c = super(Partial_PPEM, self).mStep_epsilon()
        # return self.encryption_unit.CKKS_encrypt(a), self.encryption_unit.CKKS_encrypt(b),self.encryption_unit.CKKS_encrypt(c)

        def encrypt_and_return(value):
            return self.encryption_unit.CKKS_encrypt(value)

            # Using a ThreadPoolExecutor to parallelize encryption

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit encryption tasks for a, b, and c
            a_future = executor.submit(encrypt_and_return, a)
            b_future = executor.submit(encrypt_and_return, b)
            c_future = executor.submit(encrypt_and_return, c)

            # Wait for all encryption tasks to complete
            concurrent.futures.wait([a_future, b_future, c_future])

            # Get the CKKS encrypted values from the futures
            encrypted_a = a_future.result()
            encrypted_b = b_future.result()
            encrypted_c = c_future.result()

        return encrypted_a, encrypted_b, encrypted_c
    def update(self, a_all,b_all,c_all):
        #
        # a=self.encryption_unit.decrypt(a_all)
        # b=self.encryption_unit.decrypt(b_all)
        # c=self.encryption_unit.decrypt(c_all)
        def decrypt_and_return(value):
            return self.encryption_unit.decrypt(value)

        # Using a ThreadPoolExecutor to parallelize decryption
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit decryption tasks for a_all, b_all, and c_all
            a_future = executor.submit(decrypt_and_return, a_all)
            b_future = executor.submit(decrypt_and_return, b_all)
            c_future = executor.submit(decrypt_and_return, c_all)

            # Wait for all decryption tasks to complete
            concurrent.futures.wait([a_future, b_future, c_future])

            # Get the decrypted values from the futures
            a = a_future.result()
            b = b_future.result()
            c = c_future.result()
        super(Partial_PPEM, self).update(a,b,c)
