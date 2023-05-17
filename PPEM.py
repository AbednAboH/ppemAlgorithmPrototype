import imageio as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from EM import *
from main import algortithem
import seal


class PPEM(algortithem):
    def __init__(self, n, inputType, inputDimentions, max_iter, number_of_clustures, eps=1e-4, input=None):
        super(PPEM, self).__init__(n, inputType, inputDimentions, max_iter, number_of_clustures, eps=1e-4, input=None)
        self.public_key=None
        self.private_key=None
    def encrypt_data(self,public_key, data):
        """Encrypt the data using BFV encryption"""
        context = public_key.context()
        encoder = seal.CKKSEncoder(context)
        encrypted_data = seal.Ciphertext(context)
        scale = pow(2.0, 40)
        encoder.encode(data, scale, encrypted_data)
        public_key.encrypt(encrypted_data)
        return encrypted_data

    def decrypt_data(self,private_key, encrypted_data):
        """Decrypt the data using BFV decryption"""
        context = private_key.context()
        encoder = seal.CKKSEncoder(context)
        decrypted_data = seal.Plaintext()
        private_key.decrypt(encrypted_data, decrypted_data)
        return encoder.decode(decrypted_data)
    # todo cange the signiture as you don't need this one , once the library is installed you will be able \
    #  to see if you wrote code that isn't good enough
    def eStep(self):
        """Calculate the E-step of the EM algorithm"""
        n_samples, n_features = self.n_inputs.shape
        n_components = len(self.responsibilities)

        # Calculate the probabilities of each sample belonging to each component
        log_probabilities = np.zeros((n_samples, n_components))
        for i in range(n_components):
            mean = self.means[i]
            covariance = self.covariances[i]
            responsibilities = self.responsibilities[i]
            covariance_det = np.linalg.det(covariance)
            covariance_inv = np.linalg.inv(covariance)
            coefficient = 1.0 / np.sqrt((2 * np.pi) ** n_features * covariance_det)
            for j in range(n_samples):
                x = self.responsibilities[j]
                x_encrypted = self.encrypt_data(self.public_key, x)
                mean_encrypted = self.encrypt_data(self.public_key, mean)
                covariance_inv_encrypted = self.encrypt_data(self.public_key, covariance_inv)
                quadratic_form = (x_encrypted - mean_encrypted).dot(covariance_inv_encrypted).dot(
                    x_encrypted - mean_encrypted)
                log_probabilities[j, i] = np.log(responsibilities) - 0.5 * np.log(covariance_det) - 0.5 * quadratic_form
        log_likelihood = np.sum(log_probabilities, axis=1)
        log_probabilities -= log_likelihood[:, np.newaxis]
        probabilities = np.exp(log_probabilities)
        return log_likelihood, probabilities

    def mstep(self, responsibilities, public_key, secret_key):
        # Initialize variables
        mu = []
        sigma = []
        pi = []
        epsilon = 10 ** -9
        ciphertexts = []

        # Calculate the new means, covariances, and mixing coefficients
        for i in range(self.k):
            # Calculate the new mixing coefficient
            pi_i = (np.sum(gamma[:, i]) + epsilon) / n
            pi.append(pi_i)

            # Calculate the new mean
            mu_i = np.sum(gamma[:, i].reshape(-1, 1) * responsibilities, axis=0) / (np.sum(gamma[:, i]) + epsilon)
            mu.append(mu_i)

            # Calculate the new covariance
            diff = responsibilities - mu_i
            cov_i = np.dot((gamma[:, i].reshape(-1, 1) * diff).T, diff) / (np.sum(gamma[:, i]) + epsilon)
            sigma.append(cov_i)

            # Encrypt the cluster parameters
            mu_i_encrypted = self.encrypt_data(mu_i, public_key)
            cov_i_encrypted = self.encrypt_data(cov_i, public_key)
            pi_i_encrypted = self.encrypt_data(pi_i, public_key)
            ciphertexts.append((mu_i_encrypted, cov_i_encrypted, pi_i_encrypted))

        return ciphertexts, mu, sigma, pi


if __name__ == '__main__':
    # Set number of clusters
    numberOfClusters = 3
    numberOfDimensions = 3

    # Generate toy dataset with 3 clusters
    np.random.seed(42)
    data = np.vstack((np.random.randn(100, numberOfDimensions), np.random.randn(100, numberOfDimensions) + 5,
                      np.random.randn(100, numberOfDimensions) + 10))

    # Run EM algorithm
    pi, means, covariances, log_likelihoods = em_algorithm(data, num_clusters=numberOfClusters)

    # Create meshgrid for contour plot

    twoDimentionsRepresentation(data, means[:, 1:3], covariances[:, 1:3, 1:3], numberOfClusters)

    twoDimentionsRepresentation(data, means[:, :2], covariances[:, :2, :2], numberOfClusters)

    # Plot data points and contour plot
    multiDimentionsRepresentation(data, pi, means, covariances, numberOfDimensions)
