import imageio as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from EM import *
# from encryptions import EMAlgorithm
from main import algortithem
import tenseal as ts
"""
For the parameters of the encryption model you can refer to https://github.com/OpenMined/TenSEAL/blob/main/tutorials/Tutorial%203%20-%20Benchmarks.ipynb for guidance
"""
class PPEM(algortithem):
    def __init__(self, n, inputType, inputDimentions, max_iter, number_of_clustures, eps=1e-4, input=None):
        super(PPEM, self).__init__(n, inputType, inputDimentions, max_iter, number_of_clustures, eps=1e-4, input=None)
        # create TenSEALContext
        self.context = ts.context(
            scheme=ts.SCHEME_TYPE.BFV,
            poly_modulus_degree=8192,
            plain_modulus=786433,
            coeff_mod_bit_sizes=[40, 21, 21, 21, 21, 21, 21, 40],
            encryption_type=ts.ENCRYPTION_TYPE.SYMMETRIC,)
        # scale of ciphertext to use might not work on BFV model , surly works on CKKS
        self.context.global_scale = 2 ** 40
        # dot product key needed for the dot operation
        self.context.generate_galois_keys()

    def encrypt_data(self, data):
        """Encrypt the data using BFV encryption"""
        data1=np.power(data,2)*40
        tensorData=ts.bfv_tensor(self.context,data1)
        tensorData.square().transpose_()
        return tensorData

    def decrypt_data(self,encrypted_data):
        """Decrypt the data using BFV decryption"""

        # print("decoded:\n", self.encoder.decode(decrepted))

        return encrypted_data.decrypt().tolist()


    # def homomorphic_matrixMultiplication(self,vector,matrixAsVector):
    #     ts.enc_matmul_encoding()
    def MultiVariantPDF(self,x,mean,covariance):
        #todo : to be altered to a matching function
        n = len(mean)
        coeff = 1.0 / ((2 * pi) ** (n / 2) * np.sqrt(det(covariance)))
        exponent = -0.5 * np.dot(np.dot((x - mean), inv(covariance)), (x - mean).T)
        return coeff * exp(exponent)


    def initParameters(self):
        super(PPEM, self).initParameters()
    # todo cange the signiture as you don't need this one , once the library is installed you will be able \
    #  to see if you wrote code that isn't good enough

    def eStep(self):
        """Calculate the E-step of the EM algorithm"""
        n_samples, n_features = self.n_inputs.shape
        n_components = len(self.responisbilities)
        # todo swap coefficient with pi , as you might get confused
        # Calculate the probabilities of each sample belonging to each component
        log_probabilities = np.zeros((n_samples, n_components))
        for i in range(n_components):
            mean = self.means[i]
            covariance = self.covariances[i]
            responsibilities = self.responisbilities[i]
            covariance_det = np.linalg.det(covariance)
            # todo do not use the invers , there is a better way+
            covariance_inv = np.linalg.inv(covariance)
            # print("covariance\n",covariance)
            # print("covariance inverse\n",covariance_inv)
            self.coefficient = 1.0 / np.sqrt((2 * np.pi) ** n_features * covariance_det)
            for j in range(n_samples):
                x = self.responisbilities[i]
                x_encrypted = self.encrypt_data(x)
                #encrypt parameters
                print("x_encrypted\n",self.decrypt_data(x_encrypted))
                mean_encrypted = self.encrypt_data(mean)
                print("mean_encrypted\n",self.decrypt_data(x_encrypted))
                print(5*"-------------------")
                covariance_inv_encrypted = self.encrypt_data(covariance)
                # print("inverse_Encrypted\n",self.decrypt_data(covariance_inv_encrypted))
                # responsibilities - mean
                diff_encrypted = x_encrypted.sub(mean_encrypted)
                print("minus result\n",self.decrypt_data(diff_encrypted),x-mean)

                # todo cange the way you multiply , as covariances is not actually a multi dimentional matrix
                quadratic_form_encrypted=diff_encrypted.dot(covariance_inv_encrypted)

                print("result:\n",self.decrypt_data(quadratic_form_encrypted),5*"**********"+"\n")
                # print("should be : ",np.multiply(self.decrypt_data(diff_encrypted),self.decrypt_data(covariance_inv_encrypted)),
                      # "\nactual matrix:",np.multiply((x-mean),covariance_inv))
                #
                # print(5*"**********"+"\n")
                # print(5*"-------------------")
                # print("diff_encrypted\n", self.decrypt_data(diff_encrypted))
                # print("quadric form:\n", self.decrypt_data(quadratic_form_encrypted))
                # # quadratic_form_encrypted =  diff_encrypted.dot(covariance_inv_encrypted)
                quadratic_form_encrypted =quadratic_form_encrypted.dot(diff_encrypted)
                print("result:\n" + 5 * "**********"+"\n", self.decrypt_data(quadratic_form_encrypted))
                #
                # print("should be : ",np.multiply(self.decrypt_data(quadratic_form_encrypted),self.decrypt_data(diff_encrypted)))
                # # print(5*"**********"+"\n")

                quadratic_form = self.decrypt_data(quadratic_form_encrypted)

                quadratic_form=np.array(quadratic_form)
                print("Quadric:",quadratic_form,"\n Responsibilities[i]:",responsibilities,"\ncovariance",covariance_det)

                log_probabilities[j, i] = np.log(self.responisbilities[j][i]) - 0.5 * np.log(covariance_det) - 0.5 * quadratic_form
        log_likelihood = np.sum(log_probabilities, axis=1)
        log_probabilities -= log_likelihood[:, np.newaxis]
        probabilities = np.exp(log_probabilities)
        return log_likelihood, probabilities

    def mstep(self):
        # Initialize variables
        mu = []
        sigma = []
        pi = []
        ciphertexts = []
        n_samples, n_features = self.n_inputs.shape
        # Calculate the new means, covariances, and mixing coefficients
        for i in range(self.k):
            # Calculate the new mixing coefficient
            pi_i = (np.sum(self.pi[:, i]) + self.eps) / n_samples
            pi.append(pi_i)

            # Calculate the new mean
            mu_i = np.sum(self.pi[:, i].reshape(-1, 1) * self.responisbilities, axis=0) / (np.sum(self.pi[:, i]) + self.eps)
            mu.append(mu_i)

            # Calculate the new covariance
            diff = self.responisbilities - mu_i
            cov_i = np.dot((self.pi[:, i].reshape(-1, 1) * diff).T, diff) / (np.sum(self.pi[:, i]) + self.eps)
            sigma.append(cov_i)

            # Encrypt the cluster parameters
            mu_i_encrypted = self.encrypt_data(mu_i)
            cov_i_encrypted = self.encrypt_data(cov_i)
            pi_i_encrypted = self.encrypt_data(pi_i)
            ciphertexts.append((mu_i_encrypted, cov_i_encrypted, pi_i_encrypted))
        self.pi=pi
        self.covariances=sigma
        self.means=mu
        self.ciphertexts=ciphertexts
        # return ciphertexts, mu, sigma, pi


if __name__ == '__main__':
    n = 200
    inputType = None
    inputDimentions = 2
    max_iter = 100
    number_ofClusters = 2

    pi, means, covariances, log_likelihoods = PPEM(n, inputType, inputDimentions, max_iter,
                                                          number_ofClusters).solve()
    # array= PPEM(n, inputType, inputDimentions, max_iter,
    #                                                       number_ofClusters).create_input()
    # EMAlgorithm(n,2).fit(array)
