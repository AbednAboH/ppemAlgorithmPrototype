import math
import os
import time

import imageio
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from HelpingFunctions import twoDimentionsRepresentation, twoDimentionalGifCreator
import numpy as np


class algortithem:
    """
    Basic EM algorithm

    Attributes:
        pi:to be written later
        log_likelihoods:to be written later
        covariances:to be written later
        means:to be written later
        eps:to be written later
        inputDimentions:to be written later
        tick:to be written later
        sol_time:to be written later
        max_iter:to be written later
        output:to be written later
        output2:to be written later
        iter:to be written later
        plots:to be written later
        inputParameters:to be written later
        n:to be written later
        iteration:to be written later
        numberOfSamples:to be written later
        k:to be written later
        responisbilities:to be written later
        n_inputs:to be written later
        inputDimentions:to be written later
        epsilonExceleration:to be written later
        plottingEnabled:to be written later
    """

    def __init__(self, n: int, inputDimentions: int = 2, max_iter: int = 100, number_of_clustures: int = 2,
                 eps: float = 1e-4, epsilonExceleration: bool = True, input: np.array = None,
                 plottingTools: bool = False, plot_name=""):

        """
        Initiate the algorithm's parameters

        Parameters:
            n:number of parameters.
            max_iter:number of iterations
            number_of_clustures :number of clusters to be associated with the data points
            input :input of the PPEM algorithm when we use server client model
            inputDimentions:the dimensions of the input array
            epsilonExceleration:a boolean that enables/disables convergence with epsilons aid
            eps:the value of epsilon that helps with the convergence criteria
            plottingTools:if True the algorithm plots a gif of the EM process of the algorithm
            :param Partial_EM:

        """

        self.pi = None
        self.log_likelihoods = []
        self.covariances = None
        self.means = None
        self.eps = eps
        self.inputDimentions = inputDimentions

        # data for complexity analysis
        self.tick = 0
        self.sol_time = 0
        self.max_iter = max_iter
        self.ticks = []
        self.time_line = []
        self.iter = []

        # Plots to be presented
        self.plots = []

        # todo: what is this parameter?
        self.inputParameters = list(range(n))
        # todo: define N above
        self.n = n

        # number of iterations in the em algorithm
        self.iteration = 0  # current iteration that went through the algorithm

        # number of samples /length of input
        self.numberOfSamples = n
        # number of clusters
        self.k = number_of_clustures
        # responsibilities
        self.responisbilities = None
        # array of inputs/
        np.random.seed(42)
        self.n_inputs = self.create_input() if input is None else input
        # get dimentions from data/inputs
        _, self.inputDimentions = self.n_inputs.shape

        self.epsilonExceleration = epsilonExceleration
        self.plottingEnabled = plottingTools
        self.initParameters()

        self.plot_name = plot_name



    def create_input(self):
        """
        Creates input based on the number of samples and the number of clusters(k)
        :return: np.array with all the samples created
        """
        array = []
        n_rows = int(self.numberOfSamples / self.k)
        n_cols = self.inputDimentions
        for i in range(self.k):
            if i == 0:
                array = np.random.randn(n_rows, n_cols)
                # print("\n",i,len(array))

            else:
                new_array = np.random.randn(n_rows, n_cols) + i * 5
                array = np.vstack((array, new_array))
                # print("\n",i,len(array))

        # self.n = n_rows * n_cols

        return array

    def sorting(self, population):
        # todo if you want to use it ,create a <= operator in input type
        return sorted(population, reverse=False)

    def initParameters(self):
        # todo change this
        """
        Initiates pi ,mean,covariance,responsibilities arrays
        :return: Nothing
        """
        num_samples, num_dimensions = self.n_inputs.shape
        self.pi = np.ones(self.k) / self.k
        self.means = np.random.rand(self.k, num_dimensions)
        self.covariances = np.array([np.eye(num_dimensions)] * self.k)
        self.responisbilities = np.zeros((self.numberOfSamples, self.k))

    def eStep(self):
        """
        E-step of the EM algorithm

        :return: Nothing
        """
        for j in range(self.k):
            self.responisbilities[:, j] = self.pi[j] * multivariate_normal.pdf(self.n_inputs, self.means[j],
                                                                               self.covariances[j])
        self.responisbilities /= np.sum(self.responisbilities, axis=1, keepdims=True)

    def mstep(self):
        """
        M step of the log likelyhood based EM algorithm

        :return:
        """
        # M-step: update parameters
        N_k = np.sum(self.responisbilities, axis=0)
        for j in range(self.k):
            self.means[j] = np.sum(self.responisbilities[:, j].reshape(-1, 1) * self.n_inputs, axis=0) / N_k[j]
            self.covariances[j] = np.zeros((self.inputDimentions, self.inputDimentions))
            for n in range(self.numberOfSamples):
                x = self.n_inputs[n, :] - self.means[j, :]
                self.covariances[j] += self.responisbilities[n, j] * np.outer(x, x)
            self.covariances[j] /= N_k[j]
        self.pi = N_k / self.numberOfSamples
        self.LogLikelyhood()

    def mStep_epsilon(self):
        """
        M Step of the epsilon acceleration based EM algorithm

        :return: Nothing
        """
        # M-step: update parameters
        Nk = np.sum(self.responisbilities, axis=0)
        oldMeans = self.means.copy()

        # M*-step with ε-acceleration
        for j in range(self.k):

            self.means[j] = np.sum(
                self.responisbilities[:, j].reshape(-1, 1) * self.n_inputs, axis=0
            ) / Nk[j]

            self.covariances[j] = np.zeros((self.inputDimentions, self.inputDimentions))
            for n in range(self.numberOfSamples):
                x = self.n_inputs[n, :] - self.means[j, :]
                self.covariances[j] += self.responisbilities[n, j] * np.outer(x, x)
            self.covariances[j] /= Nk[j]

            # ε-acceleration
            self.means[j] = self.eps * oldMeans[j] + (1 - self.eps) * self.means[j]

        self.pi = Nk / self.numberOfSamples

        self.LogLikelyhood()

    def getLikelyhood(self):
        return self.log_likelihoods[-1]

    def LogLikelyhood(self):
        """
        Log Likleyhood function workes on local parameters
        :return: appends LogLikeleyhood to the logLikelyhood stack
        """
        # Compute log-likelihood
        try:
            log_likelihood = np.sum(
                np.log(np.sum(self.pi[j] * multivariate_normal.pdf(self.n_inputs, self.means[j], self.covariances[j])
                              for j in range(self.k))))
            self.log_likelihoods.append(log_likelihood)

        except np.linalg.LinAlgError:
            print(f"{self.plot_name}  Singular matrix cannot be inversed! ")
            print("means\n",self.means,"\n","covariances\n",self.covariances)

    def handle_initial_time(self):
        """calculates initial time"""
        self.tick = time.time()
        self.sol_time = time.perf_counter()

    def handle_prints_time(self):
        """calculates The time it took for an iteration to complete"""
        runtime = time.perf_counter() - self.sol_time
        clockticks = time.time() - self.tick
        self.time_line.append(runtime)
        self.ticks.append(clockticks)

        print_time((runtime, clockticks))

    def algo(self, i=0):
        """
        What happens each Iteration (The Algorithm Per Iteration)

        Parameters
        ----------
        i:
            Iteration i

        Returns
        -------
        List
            list of 3 arrays and length of input, Pi,Means,Covariances,number of samples
        """

        if self.plottingEnabled: self.usePlotingTools(i, self.plot_name == "")
        self.eStep()
        return self.mStep_epsilon() if self.epsilonExceleration else self.mstep()

    def stopage(self, i):
        """Stoppage criteria

        Parameters
        ----------
        i:
            number of parameters.

        Returns
        -------
        bool
            if true then stop else continue iterating
        """

        return True if i > 1 and np.abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.eps else False

    def solve(self):
        """
        Skelaton for all algorithms derived from this class, it includes\
        iterations counter ,plot saving ,time calculations for the algorithm\


        :return: pi,means,covariances,log likelyhood
        """
        self.handle_initial_time()
        for i in range(self.max_iter):

            self.iteration += 1
            self.algo(i)

            self.iter.append(i)
            self.handle_prints_time()
            if self.stopage(i) or i == self.max_iter - 1:
                print(" number of generations : ", i)
                self.handle_prints_time()
                self.savePlotAsGif()
                self.usePlotingTools(i, True)
                break

        return self.pi, self.means, self.covariances, self.log_likelihoods, self.n_inputs,self.ticks,self.time_line

    def update_paramters(self, pi, means, covariances):
        self.pi, self.means, self.covariances = pi, means, covariances

    def usePlotingTools(self, iteration, bool):
        """Plot the EM output in 2 dimensions """
        try:
            if iteration % 4:
                if bool:
                    twoDimentionalGifCreator(self.n_inputs, self.means, self.covariances, self.k, iteration, self.plots,
                                             self.pi, self.plot_name)
                else:
                    twoDimentionalGifCreator(self.n_inputs, self.means, self.covariances, self.k, iteration, self.plots,
                                             self.pi)
        except np.linalg.LinAlgError:
            print("covariance is not Inversable or not singular")

    def savePlotAsGif(self):
        """Convert all saved plots into one GIF that shows what tranpired on the data in real time"""
        # Save the plots as a GIF
        dpi = 100
        plots = []
        dpi = 100
        if not os.path.exists('Results'):
            os.makedirs('Results')

        images = []
        for filename in self.plots:
            images.append(imageio.imread(filename))
        if self.plottingEnabled:
            imageio.mimsave(fr'Results/{self.plot_name}.gif', images, duration=200)
        self.deleteTempImages()

    def deleteTempImages(self):
        """Helper function that deletes all images created to create the GIF file"""
        for i, fig in enumerate(self.plots):
            if os.path.exists('temp'):
                # remove the file
                os.remove('temp')
        self.plots = []


# print_B = lambda x: print(f" Best:{len(x.object)} ,fittness: {x.fitness} ", end=" ")
print_B = lambda x: print(f" Best:{x} ,\nfittness: {x.fitness} ", end=" ")
# print_B = lambda x: print(f" Best: {x.object} ,fittness: {x.fitness} ", end=" ")

#  prints mean and variance
print_mean_var = lambda x: print(f"Mean: {x[0]} ,Variance: {x[1]}", end=" ")
# prints time
print_time = lambda x: print(f"Time :  {x[0]}  ticks: {x[1]}")
# calculates variance
variance = lambda x: math.sqrt((x[0] - x[1]) ** 2)

if __name__ == '__main__':
    n = 100
    inputType = None
    inputDimentions = 2
    max_iter = 1000
    number_ofClusters = 4

    pi, means, covariances, log_likelihoods, n_input,ticks,time_line = algortithem(n, inputDimentions, max_iter, number_ofClusters,
                                                                   plottingTools=True).solve()

    print(covariances)
