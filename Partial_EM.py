import imageio as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from HelpingFunctions import *
# from encryptions import EMAlgorithm
from FastEM import algortithem
from TenSEAL_encryption_unit import encryption
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


#  we can use it in two ways
#       1. either algo.solve() which calculates one iteration by default ,but also adds plotting
#       2. or just use the e/m step of each Partial EM algorithm in the server
# we went with option 2 for now , might change it in the future , but both are present in case we want to use either of them
class Partial_EM(algortithem):
    def __init__(self, n, inputDimentions: int = 2, max_iter: int = 1, number_of_clustures: int = 2, eps: float = 1e-5,
                 epsilonExceleration: bool = True,
                 input: np.array = None, plottingTools: bool = False, plot_name=""):
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
        super(Partial_EM, self).__init__(n, inputDimentions, max_iter, number_of_clustures, eps, epsilonExceleration,
                                         input, plottingTools, plot_name)

        # encryption unit for encrypting the data for each client

    def mStep_epsilon(self):
        """ calculate the sum of the responsibilities ,and the uppder side of the means equation meaning q_i,s,a* X_i then return them to the server"""
        q_i_s_a = np.sum(self.responisbilities, axis=0)
        oldMeans = self.means.copy()

        for j in range(self.k):
            self.means[j] = np.sum(self.responisbilities[:, j].reshape(-1, 1) * self.n_inputs, axis=0)
        # todo check: checked , the same as it should be
        return q_i_s_a, self.means, self.numberOfSamples

    def mStep_Covariance(self, means):
        """calculate the after you get the means from the server calculate the covariance matrix based on the means of all of the data """
        # step 2 when sending to the server !
        self.means = means
        for j in range(self.k):
            self.covariances[j] = np.zeros((self.inputDimentions, self.inputDimentions))
            for n in range(self.numberOfSamples):
                x = self.n_inputs[n, :] - self.means[j, :]
                self.covariances[j] += self.responisbilities[n, j] * np.outer(x, x)

        # print("before", self.covariances)
        return self.covariances


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

        super(Server, self).__init__(n, inputDimentions, max_iter, number_of_clustures, eps, epsilonExceleration, input,
                                     plottingTools, plot_name)
        self.clients = []
        self.partialEM = Partial_em
        self.init_clients(clients)

    def init_clients(self, num_clients):
        """ initiate the clients code the Partial EM code"""
        # 0todo create a selection method to choose data for each client for now each client creates his own data
        input_for_each_client = int(self.n / num_clients)
        np.random.shuffle(self.n_inputs)  # Shuffle the array randomly
        split_indices = np.array_split(self.n_inputs, num_clients)  # Split the indices into n sub-arrays
        # print(len(self.n_inputs))

        for i in range(num_clients):
            self.clients.append(
                self.partialEM(input_for_each_client, self.inputDimentions, 1, self.k, input=split_indices[i],
                               plot_name=self.plot_name))
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
        """
         M step
         1. apply mstep for each client and get from each client the qisa,and the sum of the upper part of the means equation
         2. calculate sum_q_i_s_a ,q_i_s_a_DOT_Xi (the upper parts of the means and Pi's  equations )
         3. update the Pi for all the data and the means of all the data
         4. send the means back to the clients and get the upper portion of the covariance equation
         5. calculate the new covariance matrices
         6. send the parameters back to the clients and calculate the log likelyhood in each clients enviroment
         7. get the likelyhood function from the clients and check for convergence
        """
        # ____________________________________________________________
        # step 1
        # ____________________________________________________________
        all_qisa, means, covariances, num_samples = [], [], [], []
        oldMeans = self.means.copy()
        # for each client do its own m step
        for client in self.clients:
            qisa, mean, numberOfSample = client.mStep_epsilon()
            all_qisa.append(qisa)
            means.append(mean)
            num_samples.append(numberOfSample)

        # ____________________________________________________________
        # step 2
        # ____________________________________________________________

        sum_q_i_s_a = np.sum(all_qisa, axis=0)
        q_i_s_a_DOT_Xi = np.sum(means, axis=0)
        # q_i_s_a_DOT_Xi_Minus_miu_squared = np.sum(covariances, axis=0)

        # ____________________________________________________________
        # step 1
        # ____________________________________________________________

        # update Pi
        self.pi = sum_q_i_s_a / np.sum(num_samples)

        # sum of clients q_(i,s,a)*Xi/sum of clients q_(i,s,a)
        self.means = q_i_s_a_DOT_Xi / sum_q_i_s_a[:, np.newaxis]

        # ____________________________________________________________
        # step 4
        # ____________________________________________________________


        for client in self.clients:
            covariances.append(client.mStep_Covariance(self.means))

        q_i_s_a_DOT_Xi_Minus_miu_squared = np.sum(covariances, axis=0)
        # q_(i, s, a)(Xi-miu_s)(Xi-miu_s)^T /sum of q_(i,s,a)
        self.covariances = q_i_s_a_DOT_Xi_Minus_miu_squared
        for j in range(self.k):
            self.covariances[j] /= sum_q_i_s_a[j]
        # ____________________________________________________________
        # step 5
        # ____________________________________________________________
        #* exceleration
        for i in range(self.k):
            self.means[i] = self.eps * oldMeans[i] + (1 - self.eps) * self.means[i]
        for client in self.clients:
            client.update_paramters(self.pi, self.means, self.covariances)
            client.LogLikelyhood()
        # ____________________________________________________________
        # step 6
        # ____________________________________________________________77

        likelyhood = [client.getLikelyhood() for client in self.clients]
        self.log_likelihoods.append(np.sum(likelyhood))

        # todo update means on all

    def stopage(self, i):
        if i > 1:
            print(np.abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]))
        return True if i > 1 and np.abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.eps and \
                       self.log_likelihoods[-1] - self.log_likelihoods[-2] >= 0 else False



if __name__ == '__main__':
    for n in range(100, 10000, 100):
        for k in range(2, 4):
            if n % k == 0:
                server = Server(n=n, max_iter=1000, number_of_clustures=k, plottingTools=False, eps=0.0001, clients=1,
                                plot_name="n300_k3_c1")
                pi, means, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()
                writeData(pi, means, covariances, log_likelihoods, n_input, ticks, time_line,
                          f"Results/MultiPartyEM/PPEM_n{n}_k3_c1")
                for clients in range(2, 10, 4):
                    if n % clients == 0 and n / k > 20:
                        print("\n\n\n", "------" * 10)
                        server = Server(n=n, max_iter=1000, number_of_clustures=k, plottingTools=False, eps=0.0001,
                                        clients=clients,
                                        plot_name=f"n{n}_k{k}_c{clients}", input=n_input)
                        pi, means, covariances, log_likelihoods, n_input, ticks, time_line = server.solve()
                        writeData(pi, means, covariances, log_likelihoods, n_input, ticks, time_line,
                                  f"Results/MultiPartyEM/PPEM_n{n}_k{k}_c{clients}")
