from Partial_EM import Partial_EM,Server
import numpy as np
from TenSEAL_encryption_unit import encryption
from HelpingFunctions import writeData
class Partial_PPEM(Partial_EM):
    def __init__(self, n, inputDimentions: int = 2, max_iter: int = 1, number_of_clustures: int = 2, eps: float = 1e-5,
                 epsilonExceleration: bool = True,
                 input: np.array = None, plottingTools: bool = False,plot_name="",encrypt=None):
        """
        Parameters:
            n:number of parameters.
            max_iter:number of iterations
            number_of_clustures :number of clusters to be associated with the data points
            input :input of the PPEM algorithm when we use server client model
            inputDimentions:the dimensions of the input array
            epsilonExceleration:a boolean that enables/disables convergence with epsilons aid
            eps:the value of epsilon that helps with the convergence criteria
            plottingTools:if True the algorithm plots a gif of the EM process of the algorithm

          """
        super(Partial_PPEM, self).__init__(n, inputDimentions, max_iter, number_of_clustures, eps, epsilonExceleration,
                                         input, plottingTools,plot_name)
        # encryption unit for encrypting the data for each client
        self.encryption_unit = encrypt
        self.qisa=None

    def update_encryption(self,context):
        self.encryption_unit=context
    def mStep_epsilon(self):
        """ calculate the sum of the responsibilities ,and the uppder side of the means equation meaning q_i,s,a* X_i then return them to the server"""
        q_i_s_a, self.means=super(Partial_PPEM, self).mStep_epsilon()
        return self.encryption_unit.CKKS_encrypt(q_i_s_a),self.encryption_unit.CKKS_encrypt(self.means)

    def mStep_Covariance(self, means,qisa=None,n=None):
        """calculate the after you get the means from the server calculate the covariance matrix based on the means of all of the data """
        # step 2 when sending to the server !
        self.means =self.encryption_unit.decrypt(means)
        self.qisa=self.encryption_unit.decrypt(qisa)

        super(Partial_PPEM, self).mStep_Covariance(self.encryption_unit.decrypt(means),
                                                   self.encryption_unit.decrypt(qisa),
                                                   self.encryption_unit.decrypt(n))

        return self.encryption_unit.CKKS_encrypt(self.covariances)

    def m_step_actualCovariances(self,covariances):
        # covariances of all the data :
        super(Partial_PPEM, self).m_step_actualCovariances(self.encryption_unit.decrypt(covariances))


class PPserver(Server):
    """
      Server EM Processor

    """

    def __init__(self, n=200, inputDimentions: int = 2, max_iter: int = 100, number_of_clustures: int = 2,
                 eps: float = 0.00001,
                 epsilonExceleration: bool = True,
                 input: np.array = None, plottingTools: bool = False, clients: int = 2,plot_name=""):
        """
        Parameters:
            n :number of parameters.
            max_iter :number of iterations
            number_of_clustures :number of clusters to be associated with the data points
            input :input of the PPEM algorithm when we use server client model
            inputDimentions:the dimensions of the input array
            epsilonExceleration:a boolean that enables/disables convergence with epsilons aid
            eps:the value of epsilon that helps with the convergence criteria
            plottingTools:if True the algorithm plots a gif of the EM process of the algorithm
            clients:The number of clients expected /created

        """

        super(PPserver, self).__init__(n, inputDimentions, max_iter, number_of_clustures, eps, epsilonExceleration, input,
                                     plottingTools,clients, plot_name,Partial_em=Partial_PPEM)

        self.encryptionUnit = encryption

    # update encryption for all clients
        self.encryptor=None
    def algo(self,i=0):
        self.encryptor=self.encryptionUnit()
        for client in self.clients:
            client.update_encryption(self.encryptor)
        super(PPserver, self).algo(i)

    def mStep_epsilon(self):
        self.n=self.encryptor.CKKS_encrypt(self.n)
        super(PPserver, self).mStep_epsilon()



    def usePlotingTools(self, iteration, bool):
        "For graph drawing functionality"
        self.covariances=self.clients[0].covariances
        self.means=self.clients[0].means
        self.pi=self.clients[0].pi
        super(PPserver, self).usePlotingTools(iteration,bool)

if __name__ == '__main__':
    for n in range(100, 10000, 100):
        for k in range(2, 4):
            if n % k == 0:
                server = PPserver(n=n, max_iter=1000, number_of_clustures=k, plottingTools=True, eps=0.0001, clients=1,
                                plot_name=f"Results/PPEM/PPEM_n{n}_k{k}_c1")
                pi, means, covariances, log_likelihoods, n_input,ticks,time_line = server.solve()
                writeData(pi, means, covariances, log_likelihoods, n_input,ticks,time_line,f"Results/PPEM/PPEM_n{n}_k3_c1.csv")
                for clients in range(2, 10, 4):
                    if n % clients == 0 and n / k > 20:
                        print("\n\n\n", "------" * 10)
                        server = PPserver(n=n, max_iter=1000, number_of_clustures=k, plottingTools=True, eps=0.0001,
                                        clients=clients,
                                        plot_name=f"Results/PPEM/PPEM_n{n}_k{k}_c{clients}", input=n_input)
                        pi, means, covariances, log_likelihoods, n_input,ticks,time_line=server.solve()
                        writeData(pi, means, covariances, log_likelihoods, n_input,ticks,time_line,f"Results/PPEM/PPEM_n{n}_k{k}_c{clients}.csv")
