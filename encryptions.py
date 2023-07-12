import numpy as np

# Example data
from scipy.stats import multivariate_normal

data = [
    [1.2, 2.3],
    [0.8, 1.7],
    [1.6, 3.2],
    [1.0, 2.0]
]

# Number of nodes
num_nodes = len(data)

# Initialize parameters
K = 2  # Number of components in the GMM
D = len(data[0])  # Dimension of the data
pi = np.ones(K) / K  # Mixing coefficients
mu = np.random.randn(K, D)  # Means
cov = np.zeros((K, D, D))  # Covariance matrices
for k in range(K):
    cov[k] = np.eye(D)

# Federated EM algorithm
num_iterations = 10
for t in range(num_iterations):
    # Local updates at each node
    a_t = []
    b_t = []
    c_t = []

    # Perform local E-step and M-step at each node
    for i in range(num_nodes):
        node_data = data[i]
        N = len(node_data)

        # E-step
        gamma_tij = np.zeros((N, K))
        for n in range(N):
            x = node_data[n]
            for k in range(K):
                gamma_tij[n, k] = pi[k] * multivariate_normal.pdf(x, mean=mu[k], cov=cov[k])
            gamma_tij[n] /= np.sum(gamma_tij[n])

        # M-step
        a_tij = gamma_tij.sum(axis=0) / N
        b_tij = np.dot(gamma_tij.T, node_data)
        c_tij = np.zeros((K, D, D))
        for k in range(K):
            for n in range(N):
                x_minus_mu = np.reshape(node_data[n] - mu[k], (D, 1))
                c_tij[k] += gamma_tij[n, k] * np.dot(x_minus_mu, x_minus_mu.T)
        c_tij /= np.sum(gamma_tij, axis=0)[:, np.newaxis]

        a_t.append(a_tij)
        b_t.append(b_tij)
        c_t.append(c_tij)

    # Server aggregation
    a_tj = np.sum(a_t, axis=0)
    b_tj = np.sum(b_t, axis=0)
    c_tj = np.sum(c_t, axis=0)

    pi = a_tj / num_nodes
    mu = b_tj / a_tj[:, np.newaxis]
    cov = c_tj / a_tj[:, np.newaxis, np.newaxis]

# Print the learned GMM parameters
print("pi:", pi)
print("mu:", mu)
print("cov:", cov)

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_gmm_clustering(data, pi, mu, cov):
    # Plot data points
    plt.scatter(data[:, 0], data[:, 1], color='b', alpha=0.5)

    # Plot GMM components
    for k in range(len(pi)):
        # Plot ellipse representing the covariance matrix
        eigvals, eigvecs = np.linalg.eigh(cov[k])
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * np.sqrt(2 * eigvals)
        ell = Ellipse(xy=mu[k], width=width, height=height, angle=angle, color='r', alpha=0.3)
        ell.set_facecolor('none')
        plt.gca().add_patch(ell)

        # Plot mean of the component
        plt.scatter(mu[k, 0], mu[k, 1], marker='x', color='r')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('GMM Clustering')
    plt.show()
plot_gmm_clustering(data,pi,mu,cov)