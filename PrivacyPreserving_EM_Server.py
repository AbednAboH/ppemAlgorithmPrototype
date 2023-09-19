from PrivacyPreserving_EM_Client import Partial_PPEM
from Settings_Parameters import PERSESSION, MAXITER
from TenSEAL_encryption_unit import encryption
from HelpingFunctions import writeData, colored_plot
from Federated_EM_Server import Server
from PrivacyPreserving_EM_Client import Partial_PPEM
import numpy as np

class PPserver(Server):
    """
      Server EM Processor

    """

    def __init__(self, n=200, inputDimentions: int = 2, max_iter: int = 100, number_of_clustures: int = 2,
                 eps: float = 0.00001,
                 epsilonExceleration: bool = True,
                 input: np.array = None, plottingTools: bool = False, clients: int = 2,plot_name="",show_time=False):
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
                                       ,clients=clients, plot_name=plot_name,Partial_em=Partial_PPEM,coloring_feature=None,show_time=show_time)
        self.name="Privacy Preserving Expectation Maximization"
        self.encryptionUnit = encryption

    def update_all_clients(self, a, b, c,n=None):
        super(PPserver, self).update_all_clients(a,b,c,n)
        

    # update encryption for all clients
        self.encryptor=None
    def algo(self,i=0):
        self.encryptor=self.encryptionUnit()

        for client in self.clients:
            client.update_encryption(self.encryptor)
        super(PPserver, self).algo(i)


    def usePlotingTools(self, iteration, bool):
        "For graph drawing functionality"
        self._covariances=self.clients[0]._covariances
        self._means=self.clients[0]._means
        self._pi=self.clients[0]._pi
        super(PPserver, self).usePlotingTools(iteration,True)


def PP_EM_Tests():
    for n in range(200 , 10000, 900):
        for k in range(2, 4,1):

                print(f"\n generating {n} Data with {k} Gaussian mixture models \n")
                server = PPserver(n=n, max_iter=1, number_of_clustures=k, plottingTools=False, eps=PERSESSION, clients=1,
                                  plot_name=f"Results/PPEM/PPEM_n{n}_k{k}_c1")
                pi, means, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()

                for clients in range(2, 11, 4):
                    print("\n\n\n", "------" * 10,
                          f"testing with {clients} Clients with {n} data points and {k} Gaussian distributions")

                    server = PPserver(n=n, max_iter=MAXITER, number_of_clustures=k, plottingTools=False, eps=PERSESSION,
                                          clients=clients,
                                          plot_name=f"Results/PPEM/PPEM_n{n}_k{k}_c{clients}", input=n_input)
                    pi, means, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()
                    writeData(pi, means, covariances, log_likelihoods, n_input, ticks, time_line,
                                  f"Results/PPEM/Charts/PPEM_n{n}_k{k}_c{clients}.csv")
                    if k==2:
                        colored_plot(n_input, means, covariances, pi,
                                  f"Results/PPEM/PPEM_n{n}_k{k}_c{clients}_colored")

PP_EM_Tests()