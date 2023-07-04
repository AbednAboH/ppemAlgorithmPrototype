import math
import os
import random
import time

import imageio
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from EM import twoDimentionsRepresentation, twoDimentionalGifCreator

import numpy as np

from settings import GA_MAXITER

import numpy

from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

"""
         basic EM 

        @param n: number of parameters.
        @param inputType: type of input cc data.
        @param max_iter: number of iterations
        @param number_of_clustures :number of clusters to be associated with the data points 
        @param input : input of the PPEM algorithm when we use server client model 
        @return: A string greeting the person.
        for normal EM with likelyhood function algorithm use eps=0 
"""


class algortithem:

    def __init__(self, n, inputType, inputDimentions, max_iter, number_of_clustures, eps=1e-4, epsilonExceleration=True,
                 input=None, plottingTools=False):

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
        self.output = []
        self.output2 = []
        self.iter = []

        # Plots to be presented
        self.plots = []

        # todo: what is this parameter?
        self.inputParameters = list(range(n))
        # todo: define N above
        self.n = n

        # number of iterations in the em algorithm
        self.iteration = 0  # current iteration that went through the algorithm
        # todo: maybe just remove the type of the input variable
        self.inputType = inputType
        # number of samples /length of input
        self.numberOfSamples = n
        # number of clusters
        self.k = number_of_clustures
        # todo: might just change it
        self.distributions = []
        # responsibilities
        self.responisbilities = None
        # array of inputs/
        np.random.seed(42)
        self.n_inputs = self.create_input() if input == None else input
        # get dimentions from data/inputs
        _, self.inputDimentions = self.n_inputs.shape

        self.epsilonExceleration = epsilonExceleration
        self.plottingEnabled = plottingTools
        self.initParameters()

    def create_input(self):
        array = None
        n_rows = 100
        n_cols = self.inputDimentions
        for i in range(self.k):
            if i == 0:
                array = np.random.randn(n_rows, n_cols)
            else:
                new_array = np.random.randn(n_rows, n_cols) + i * 5
                array = np.vstack((array, new_array))
        self.n = n_rows * n_cols
        return array

    def sorting(self, population):
        # todo if you want to use it ,create a <= operator in input type
        return sorted(population, reverse=False)

    def initParameters(self):
        # todo change this
        num_samples, num_dimensions = self.n_inputs.shape
        self.pi = np.ones(self.k) / self.k
        self.means = np.random.rand(self.k, num_dimensions)
        self.covariances = np.array([np.eye(num_dimensions)] * self.k)
        self.responisbilities = np.zeros((self.numberOfSamples, self.k))

    def eStep(self):

        for j in range(self.k):
            self.responisbilities[:, j] = self.pi[j] * multivariate_normal.pdf(self.n_inputs, self.means[j],
                                                                               self.covariances[j])
        self.responisbilities /= np.sum(self.responisbilities, axis=1, keepdims=True)

    def mstep(self):
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

    def LogLikelyhood(self):
        # Compute log-likelihood
        log_likelihood = np.sum(
            np.log(np.sum(self.pi[j] * multivariate_normal.pdf(self.n_inputs, self.means[j], self.covariances[j])
                          for j in range(self.k))))
        self.log_likelihoods.append(log_likelihood)

    def handle_initial_time(self):
        self.tick = time.time()
        self.sol_time = time.perf_counter()

    def handle_prints_time(self):
        runtime = time.perf_counter() - self.sol_time
        clockticks = time.time() - self.tick

        print_time((runtime, clockticks))

    def algo(self, i=0):
        if self.plottingEnabled: self.usePlotingTools(i)
        self.eStep()
        return self.mStep_epsilon() if self.epsilonExceleration else self.mstep()

    def stopage(self, i):
        return True if i > 1 and np.abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.eps else False

    def solve(self):
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
                break

        return self.pi, self.means, self.covariances, self.log_likelihoods

    def usePlotingTools(self, iteration):
        twoDimentionalGifCreator(self.n_inputs, self.means, self.covariances, self.k, iteration, self.plots, self.pi)

    def savePlotAsGif(self):
        # Save the plots as a GIF
        dpi = 100
        plots = []
        for i, fig in enumerate(self.plots):
            if not os.path.exists('temp'):
                os.makedirs('temp')

            fig.savefig('temp/temp%d.png' % i, dpi=dpi)
            plt.close(fig)
            plots.append(imageio.imread('temp/temp%d.png' % i))
        if not os.path.exists('Results'):
            os.makedirs('Results')
        imageio.mimsave('Results/PlotOfClustures.gif', plots, fps=5)
        self.deleteTempImages()

    def deleteTempImages(self):
        for i, fig in enumerate(self.plots):
            if os.path.exists('temp/temp%d.png' % i):
                # remove the file
                os.remove('temp/temp%d.png' % i)


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
    n = 300
    inputType = None
    inputDimentions = 2
    max_iter = 100
    number_ofClusters = 3

    pi, means, covariances, log_likelihoods = algortithem(n, inputType, inputDimentions, max_iter,
                                                          number_ofClusters).solve()
