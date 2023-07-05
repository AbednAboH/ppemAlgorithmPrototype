import imageio as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from EM import *
# from encryptions import EMAlgorithm
from FastEM import algortithem
from TenSEAL_encryption_unit import encryption


#  we can use it in two ways
#       1. either algo.solve() which calculates one iteration by default ,but also adds plotting
#       2. or just use the e/m step of each Partial EM algorithm in the server
# we went with option 2 for now , might change it in the future , but both are present in case we want to use either of them
class Partial_EM(algortithem):
    def __init__(self, n, inputDimentions: int = 2, max_iter: int = 1, number_of_clustures: int = 2, eps: float = 1e-5,
                 epsilonExceleration: bool = True,
                 input: np.array = None, plottingTools: bool = False):
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
                                         input, plottingTools)

        # encryption unit for encrypting the data for each client
        self.encryption_unit = None
        # q from i to a

    def mStep_epsilon(self):
        """ calculate the sum of the responsibilities ,and the uppder side of the means equation meaning q_i,s,a* X_i then return them to the server"""
        q_i_s_a = np.sum(self.responisbilities, axis=0)
        oldMeans = self.means.copy()

        for j in range(self.k):
            self.means[j] = np.sum(self.responisbilities[:, j].reshape(-1, 1) * self.n_inputs, axis=0)

        return q_i_s_a, self.means, self.numberOfSamples

    def mStep_Covariance(self, means):
        """calculate the after you get the means from the server calculate the covariance matrix based on the means of all of the data """
        # step 2 when sending to the server !
        for j in range(self.k):
            self.covariances[j] = np.zeros((self.inputDimentions, self.inputDimentions))
            for n in range(self.numberOfSamples):
                x = self.n_inputs[n, :] - means[j, :]
                self.covariances[j] += self.responisbilities[n, j] * np.outer(x, x)

        print("before", self.covariances)
        return self.covariances

    def stopage(self, i):
        return True if np.abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.eps else False


class Server(algortithem):
    """
      Server EM Processor

    """

    def __init__(self, n=200, inputDimentions: int = 2, max_iter: int = 100, number_of_clustures: int = 2,
                 eps: float = 1e-5,
                 epsilonExceleration: bool = True,
                 input: np.array = None, plottingTools: bool = False, clients: int = 2):
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

        super(Server, self).__init__(n, inputDimentions, max_iter, number_of_clustures, eps,
                                     epsilonExceleration,
                                     input, plottingTools)
        self.clients = []
        self.init_clients(clients)
        # for future use :
        self.encryptionUnit = encryption

    def init_clients(self, num_clients):
        """ initiate the clients code the Partial EM code"""
        # todo create a selection method to choose data for each client for now each client creates his own data
        input_for_each_client = int(n / num_clients)
        for i in range(num_clients):
            self.clients.append(Partial_EM(input_for_each_client, self.inputDimentions, 1, self.k))
            # todo:for now, in case you change the way the input is created you have to change these lines or delete them
            if i == 0:
                self.n_inputs = self.clients[i]
            else:
                self.n_inputs = np.vstack((self.n_inputs, self.clients[i]))

    def eStep(self):
        """send instructions to the clients to calculate their own e-step on their data"""
        for client in self.clients:
            client.eStep()

    def mStep_epsilon(self):
        """
         M step
         1. apply mstep for each client and get from each client the qisa,and the sum of the upper part of the means equation
         2. calculate all_q_i_s_a ,q_i_s_a_DOT_Xi (the upper parts of the means and Pi's  equations )
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

        # for each client do its own m step
        for client in self.clients:
            qisa, mean, numberOfSample = client.mStep_epsilon()
            all_qisa.append(qisa)
            means.append(mean)
            num_samples.append(numberOfSample)

        # ____________________________________________________________
        # step 2
        # ____________________________________________________________


        all_q_i_s_a = np.sum(all_qisa, axis=0)
        q_i_s_a_DOT_Xi = np.sum(means, axis=0)
        # q_i_s_a_DOT_Xi_Minus_miu_squared = np.sum(covariances, axis=0)

        # ____________________________________________________________
        # step 1
        # ____________________________________________________________

        # update Pi
        self.pi = all_q_i_s_a / np.sum(num_samples)
        # sum of clients q_(i,s,a)*Xi/sum of clients q_(i,s,a)
        self.means = q_i_s_a_DOT_Xi / all_q_i_s_a

        # ____________________________________________________________
        # step 4
        # ____________________________________________________________

        for client in self.clients:
            covariances.append(client.mStep_Covariance(self.means))

        q_i_s_a_DOT_Xi_Minus_miu_squared = np.sum(covariances, axis=0)
        # q_(i, s, a)(Xi-miu_s)(Xi-miu_s)^T /sum of q_(i,s,a)
        self.covariances = q_i_s_a_DOT_Xi_Minus_miu_squared / all_q_i_s_a

        # ____________________________________________________________
        # step 5
        # ____________________________________________________________

        for client in self.clients:
            client.update_paramters(self.pi, self.means, self.covariances)
            client.LogLikelyhood()
        # ____________________________________________________________
        # step 6
        # ____________________________________________________________

        likelyhood = [client.getLikelyhood() for client in self.clients]
        self.log_likelihoods.append(np.mean(likelyhood))


        # todo add epsilon emplementation for faster convergence
        # self.means = np.array(self.eps * oldMeans[j] + (1 - self.eps) * self.means[j])


if __name__ == '__main__':
    n = 1000
    server = Server(n=n)
    server.solve()
