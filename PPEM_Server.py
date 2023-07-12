from PPEM import Partial_PPEM
from TenSEAL_encryption_unit import encryption
from HelpingFunctions import writeData
from EMserver import Server
from PPEM import Partial_PPEM
import numpy as np

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

        super(PPserver, self).__init__(n=n, inputDimentions=inputDimentions
                                       , max_iter=max_iter,
                                       number_of_clustures=number_of_clustures
                                       , eps=eps,
                                       epsilonExceleration=epsilonExceleration,
                                       input=input,
                                     plottingTools=plottingTools
                                       ,clients=clients, plot_name=plot_name,Partial_em=Partial_PPEM)

        self.encryptionUnit = encryption

    # update encryption for all clients
        self.encryptor=None
    def algo(self,i=0):
        self.encryptor=self.encryptionUnit()

        for client in self.clients:
            client.update_encryption(self.encryptor)
        super(PPserver, self).algo(i)


    def usePlotingTools(self, iteration, bool):
        "For graph drawing functionality"
        self._covariances=self.clients[0].covariances_
        self._means=self.clients[0].means_
        self._pi=self.clients[0].pi
        super(PPserver, self).usePlotingTools(iteration,True)

if __name__ == '__main__':
    for n in range(100, 10000, 100):
        for k in range(2, 4):
            if n % k == 0:
                server = PPserver(n=n, max_iter=1000, number_of_clustures=k, plottingTools=False, eps=0.0001, clients=1,
                                plot_name=f"Results/PPEM/PPEM_n{n}_k{k}_c1")
                pi, means, covariances, log_likelihoods, n_input,ticks,time_line = server.solve()
                writeData(pi, means, covariances, log_likelihoods, n_input,ticks,time_line,f"Results/PPEM/PPEM_n{n}_k3_c1.csv")
                for clients in range(2, 10, 4):
                    if n % clients == 0 and n / k > 20:
                        print("\n\n\n", "------" * 10)
                        server = PPserver(n=n, max_iter=1000, number_of_clustures=k, plottingTools=False, eps=0.0001,
                                        clients=clients,
                                        plot_name=f"Results/PPEM/PPEM_n{n}_k{k}_c{clients}", input=n_input)
                        pi, means, covariances, log_likelihoods, n_input,ticks,time_line=server.solve()
                        writeData(pi, means, covariances, log_likelihoods, n_input,ticks,time_line,f"Results/PPEM/PPEM_n{n}_k{k}_c{clients}.csv")
