import numpy as np

from HelpingFunctions import *
from FastEM import algortithem
from Partial_EM import Partial_EM


class Server(algortithem):
    """
      Server EM Processor

    """

    def __init__(self, n=200, inputDimentions: int = 2, max_iter: int = 100, number_of_clustures: int = 2,
                 eps: float = 0.00001,
                 epsilonExceleration: bool = True,
                 input: np.array = None, plottingTools: bool = False, clients: int = 2, plot_name="",
                 Partial_em=Partial_EM):
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
                                     plottingTools=plottingTools, plot_name=plot_name)
        self.clients = []
        self.partialEM = Partial_em
        self.init_clients(clients)

    def init_clients(self, num_clients):
        """ initiate the clients code the Partial EM code"""
        # 0todo create a selection method to choose data for each client for now each client creates his own data
        input_for_each_client = int(self.n / num_clients)
        np.random.shuffle(self.n_inputs)  # Shuffle the array randomly
        split_indices = np.array_split(self.n_inputs, num_clients)  # Split the indices into n sub-arrays

        for i in range(num_clients):
            self.clients.append(
                self.partialEM(n=input_for_each_client, inputDimentions=self.inputDimentions
                               , max_iter=1, number_of_clustures=self.k, input=split_indices[i],
                               plot_name=self.plot_name, eps=self.eps))
            # todo:for now, in case you change the way the input is created you have to change these lines or delete them
            # if i == 0:
            #     self.n_inputs = self.clients[i]
            # else:
            #     self.n_inputs = np.vstack((self.n_inputs, self.clients[i]))

    def eStep(self):
        """send instructions to the clients to calculate their own e-step on their data"""
        for client in self.clients:
            client.eStep()

    def mStep_epsilon(self):


        all_qisa, means, covariances, num_samples = [], [], [], []
        # for each client do its own m step
        for client in self.clients:
            qisa, mean, cov = client.mStep_epsilon()
            all_qisa.append(qisa)
            means.append(mean)
            covariances.append(cov)

        a = np.sum(all_qisa, axis=0)
        b = np.sum(means, axis=0)
        c = np.sum(covariances, axis=0)

        self.update_all_clients(a, b, c)

        self._pi=self.clients[0]._pi
        self._covariances=self.clients[0]._covariances
        self._means=self.clients[0]._means


        likelyhood = [client.getLikelyhood() for client in self.clients]
        self.log_likelihoods.append(np.sum(likelyhood))

    def update_all_clients(self, a, b, c,n=None):
        if n==None:
            n=self.n
        for index, client in enumerate(self.clients):
            client.update(a.copy(), b.copy(), c.copy(), n)
            client.LogLikelyhood()

    def stopage(self, i):
        # return False
        return True if i > 1 and np.abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.eps else False


if __name__ == '__main__':
    for n in range(100, 10000, 100):
        for k in range(2, 4):
            if n % k == 0:
                server = Server(n=n, max_iter=1000, number_of_clustures=k, plottingTools=False, eps=0.0001, clients=1,
                                plot_name=f"Results/MultiPartyEM/EM_n{n}_k{k}_c1")
                pi, means, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()
                writeData(pi, means, covariances, log_likelihoods, n_input, ticks, time_line,
                          f"Results/MultiPartyEM/EM_n{n}_k3_c1")
                for clients in range(2, 10, 4):
                    if n % clients == 0 and n / k > 20:
                        print("\n\n\n", "------" * 10)
                        server = Server(n=n, max_iter=1000, number_of_clustures=k, plottingTools=False, eps=0.0001,
                                        clients=clients,
                                        plot_name=f"n{n}_k{k}_c{clients}", input=n_input)
                        pi, means, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()
                        writeData(pi, means, covariances, log_likelihoods, n_input, ticks, time_line,
                                  f"Results/MultiPartyEM/EM_n{n}_k{k}_c{clients}")
