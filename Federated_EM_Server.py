import random
import threading

import numpy as np

from HelpingFunctions import *
from Expectation_Maximization import algortithem
from Federated_EM_Client import Partial_EM
import numpy as np
from entropy_estimators import entropy, mi
from Settings_Parameters import FEM, PERSESSION, MAXITER
from sklearn.feature_selection import mutual_info_regression

import concurrent.futures
class Server(algortithem):
    """
      Server EM Processor

    """

    def __init__(self, n=200, inputDimentions: int = 2, max_iter: int = 100, number_of_clustures: int = 2,
                 eps: float = 0.00001,
                 epsilonExceleration: bool = True,
                 input: np.array = None, plottingTools: bool = False, clients: int = 2, plot_name="",
                 Partial_em=Partial_EM,coloring_feature=None,check_mi=False,show_time=False):
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

        super(Server, self).__init__(n=n, inputDimentions=inputDimentions, max_iter=max_iter,
                                     number_of_clustures=number_of_clustures
                                     , eps=eps
                                     , epsilonExceleration=epsilonExceleration, input=input,
                                     plottingTools=plottingTools, plot_name=plot_name,show_time=show_time)
        self.coloring_feature=coloring_feature
        self.clients = []
        self.partialEM = Partial_em
        self.init_clients(clients)
        self.name=FEM
        self.check_mi=check_mi
        self.mutual_information=[]


    def init_clients(self, num_clients):
        """ initiate the clients code the Partial EM code"""
        # 0todo create a selection method to choose data for each client for now each client creates his own data
        input_for_each_client = int(self.n / num_clients)
        np.random.shuffle(self.n_inputs)  # Shuffle the array randomly
        split_indices = np.array_split(self.n_inputs, num_clients)  # Split the indices into n sub-arrays
        # if self.n%2 !=0:
        #     split_indices[1]=split_indices[1]+[split_indices[0][random.randint(0,len(split_indices-1))]]
        for i in range(num_clients):
            self.clients.append(
                self.partialEM(n=len(split_indices[i]), inputDimentions=self.inputDimentions
                               , max_iter=self.max_iter, number_of_clustures=self.k, input=split_indices[i],
                               plot_name=self.plot_name, eps=self.eps))




    def eStep(self):
        """send instructions to the clients to calculate their own e-step on their data"""
        for client in self.clients:
            client.eStep()

    def mStep_epsilon(self):



        a_i, b_i, c_i, num_samples = [], [], [], []

        # get all relevant information from clients
        for client in self.clients:
            qisa, mean, cov = client.mStep_epsilon()
            a_i.append(qisa)
            b_i.append(mean)
            c_i.append(cov)



        # sumation of those elements
        a = np.sum(a_i, axis=0)
        b = np.sum(b_i, axis=0)
        c = np.sum(c_i, axis=0)
        self.update_all_clients(a, b, c)

        # get  the likelyhood from each client
        likelyhood = [client.getLikelyhood() for client in self.clients]
        self.log_likelihoods.append(np.sum(likelyhood))



        # mutual information estimation for inspection purposes only ,
        if self.check_mi:
            if FEM not in self.name:
                A=self.clients[0].encryption_unit.decrypt(a)
                B=self.clients[0].encryption_unit.decrypt(b)
                self.mutual_information.append(mi(np.array(A[:-1]),B,k=self.inputDimentions-1))


        # these are just for the purpose of displaying the end result of the em algorithm
        self._pi=self.clients[0]._pi
        self._covariances=self.clients[0]._covariances
        self._means=self.clients[0]._means




    def update_all_clients(self, a, b, c,n=None):
        if n is None:
            n = self.n
            # Using a ThreadPoolExecutor to parallelize the updates
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Assuming you have a list of clients in self.clients, you can iterate over them
                futures = [executor.submit(self.update_client, client, a.copy(), b.copy(), c.copy()) for client in
                           self.clients]

                # Wait for all tasks to complete
                concurrent.futures.wait(futures)

    def update_client(self, client, a, b, c):
        # Call the client's update and LogLikelyhood methods here
        client.update(a, b, c)
        client.LogLikelyhood()
    def stopage(self, i):
        # return False
        return True if i > 1 and np.abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.eps else False


def testsFEM():
    for n in range(200, 10000, 900):
        for k in range(2, 7, 1):
            print(f"\n generating {n} Data with {k} Gaussian mixture models \n")
            server = Server(n=n, max_iter=1, number_of_clustures=k, plottingTools=False, eps=PERSESSION, clients=1,
                            plot_name=f"Results/FEM/EM_n{n}_k{k}_c1")
            pi, means, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()
            for clients in range(2, 11, 4):
                try:
                    print("\n\n\n", "------" * 10,f"testing with {clients} Clients with {n} data points and {k} Gaussian distributions")
                    server = Server(n=n, max_iter=MAXITER, number_of_clustures=k, plottingTools=False, eps=PERSESSION,
                                    clients=clients,
                                    plot_name=f"Results/FEM/EM_n{n}_k{k}_c{clients}", input=n_input)
                    pi, means, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()
                    writeData(pi, means, covariances, log_likelihoods, n_input, ticks, time_line,
                              f"Results/FEM/Charts/EM_n{n}_k{k}_c{clients}")
                    if k==2:
                        colored_plot(n_input, means, covariances, pi,
                                 f"Results/FEM/EM_n{n}_k{k}_c{clients}_colored")
                except Exception as e:
                    print(e.args)
                    continue



