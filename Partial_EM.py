import imageio as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from EM import *
# from encryptions import EMAlgorithm
from FastEM import algortithem
from TenSEAL_encryption_unit import encryption
from epydoc import *

class Partial_EM(algortithem):
    def __init__(self, n, inputDimentions: int = 2, max_iter:int=1, number_of_clustures:int=2, eps:float=1e-4,
                 epsilonExceleration:bool=True,
                 input:np.array=None, plottingTools:bool=False):
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

    def mStep_epsilon(self):
        # Initialize variables
        n_samples, n_features = self.n_inputs.shape
        self.pi = np.sum(self.responisbilities, axis=0)
        oldMeans = self.means.copy()
        # Calculate the new means, covariances, and mixing coefficients

        # todo : check
        # pi_mid=np.sum(self.responisbilities, axis=0)
        # mu_mid =[ np.sum(self.pi[:, i].reshape(-1, 1) * self.responisbilities[i], axis=0) for i in range(self.k)]
        # cov_mid =[np.dot((self.pi[:, i].reshape(-1, 1) * self.responisbilities[i] - mu_mid[i]).T, self.responisbilities[i] - mu_mid[i]) for i in range(self.k)]
        # todo : endCheck
        means = []
        covariances = []
        for j in range(self.k):
            means[j] = np.sum(self.responisbilities[:, j].reshape(-1, 1) * self.n_inputs, axis=0)
            covariances[j] = np.zeros((self.inputDimentions, self.inputDimentions))
            for n in range(self.numberOfSamples):
                x = self.n_inputs[n, :] - means[j, :]
                covariances[j] += self.responisbilities[n, j] * np.outer(x, x)
        return self.pi, means, covariances, self.numberOfSamples

    def updateM_step(self, pi, means, covariance):
        self.pi = pi
        self.means = means
        self.covariances = covariance


class Server(algortithem):
    """
      Server EM Processor

    """
    def __init__(self, n, inputDimentions: int = 2, max_iter:int=1, number_of_clustures:int=2, eps:float=1e-4,
                 epsilonExceleration:bool=True,
                 input:np.array=None, plottingTools:bool=False, clients:int=2):
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
        self.init_clients(clients)
        ''' '''
        self.clients = []
        # for future use :
        self.encryptionUnit = encryption

    def init_clients(self, num_clients):
        # todo create a selection method to choose data for each client
        for _ in range(num_clients):
            self.clients.append(Partial_EM(self.n, self.inputDimentions, 1, self.k))

    def eStep(self):
        for client in self.clients:
            client.eStep()

    def mStep_epsilon(self):
        pis, means, covariances, num_samples = [], [], [], []

        for client in self.clients:
            pi, mean, covariance, numberOfSample = client.mStep_epsilon()
            pis.append(pi)
            means.append(mean)
            covariances.append(covariance)
            num_samples.append(numberOfSample)
        # should be turned to dimentional sum
        # meaning adding a for loop to go over the correct axis

        q_i_s_a = np.sum(pis, axis=0)
        q_i_s_a_DOT_Xi = np.sum(means, axis=0)
        q_i_s_a_DOT_Xi_Minus_miu_squared = np.sum(covariances, axis=0)

        self.pi = q_i_s_a / np.sum(num_samples)
        # sum of clients q_(i,s,a)*Xi/sum of clients q_(i,s,a)
        self.means = q_i_s_a_DOT_Xi / q_i_s_a[:, np.newaxis]
        # q_(i, s, a)(Xi-miu_s)(Xi-miu_s)^T /sum of q_(i,s,a)
        self.covariances = q_i_s_a_DOT_Xi_Minus_miu_squared / q_i_s_a[:, np.newaxis]
        # if this is done then we can say that the algorithm is done , what remains is to use the encryption module from the code

        # todo add epsilon emplementation for faster convergence
        # self.means = np.array(self.eps * oldMeans[j] + (1 - self.eps) * self.means[j])


if __name__ == '__main__':
    n = 200
    inputType = None
    inputDimentions = 2
    max_iter = 100
    number_ofClusters = 2

    pi, means, covariances, log_likelihoods = Partial_EM(n, inputType, inputDimentions, max_iter,
                                                         number_ofClusters).solve()
    # array= PPEM(n, inputType, inputDimentions, max_iter,
    #                                                       number_ofClusters).create_input()
    # EMAlgorithm(n,2).fit(array)
