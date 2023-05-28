import imageio as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from EM import *
# from encryptions import EMAlgorithm
from main import algortithem
import seal


class PPEM(algortithem):
    def __init__(self, n, inputType, inputDimentions, max_iter, number_of_clustures, eps=1e-4, input=None):
        super(PPEM, self).__init__(n, inputType, inputDimentions, max_iter, number_of_clustures, eps=1e-4, input=None)
        params = seal.EncryptionParameters(seal.scheme_type.bfv)
        params.set_poly_modulus_degree(4096)
        params.set_coeff_modulus(seal.CoeffModulus.BFVDefault(4096))
        params.set_plain_modulus(40961)
        self.context = seal.SEALContext(params)
        self.keygen=seal.KeyGenerator(self.context)
        # self.relin_keys = self.keygen.create_relin_keys()
        self.ciphertexts = None
        self.public_key=self.keygen.create_public_key()
        self.private_key=self.keygen.secret_key()
        self.encoder=seal.BatchEncoder(self.context)
        self.evaluate=seal.Evaluator(self.context)
    def encrypt_data(self, data):
        """Encrypt the data using BFV encryption"""
        # Create a plaintext object and encode the data
        # data is scaled up because the encoder taked only integers from type numpy.int64
        data1=np.power(data,2)*40
        data1=data1.astype(np.int64)
        data1=np.ravel(data1)
        # print("orig:\n",data)
        # print("raveled and powered up:\n",data1)
        data1=self.encoder.encode(data1)

        # Create an encryptor using the public key
        encryptor = seal.Encryptor(self.context, self.public_key)

        # Encrypt the plaintext using the encryptor
        encrypted_data=encryptor.encrypt(data1)
        # print("encrypted:\n",encrypted_data)
        # print("encrypted:\n",self.decrypt_data(encrypted_data))

        return encrypted_data

    def decrypt_data(self,encrypted_data):
        """Decrypt the data using BFV decryption"""
        decreptor = seal.Decryptor(self.context, self.private_key)
        decrepted = decreptor.decrypt(encrypted_data)
        # print("decoded:\n", self.encoder.decode(decrepted))

        # data1=np.power(self.encoder.decode(decrepted),1/2)/40

        return self.encoder.decode(decrepted)

    def initParameters(self):
        super(PPEM, self).initParameters()
        self.covariances=np.ones()
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
            covariance_inv = np.linalg.inv(covariance)
            print("covariance\n",covariance)
            print("covariance inverse\n",covariance_inv)
            self.coefficient = 1.0 / np.sqrt((2 * np.pi) ** n_features * covariance_det)
            for j in range(n_samples):
                x = self.responisbilities[j]
                x_encrypted = self.encrypt_data(x)
                print("x_encrypted\n",self.decrypt_data(x_encrypted))
                mean_encrypted = self.encrypt_data(mean)
                print("mean_encrypted\n",self.decrypt_data(x_encrypted))
                print(5*"-------------------")
                covariance_inv_encrypted = self.encrypt_data(covariance_inv)
                print("inverse_Encrypted\n",self.decrypt_data(covariance_inv_encrypted))
                diff_encrypted = self.evaluate.sub(x_encrypted , mean_encrypted)
                print("minus result\n",self.decrypt_data(diff_encrypted),x-mean)

                quadratic_form_encrypted=self.evaluate.multiply(diff_encrypted, covariance_inv_encrypted)

                print("result:\n"+5*"**********"+"\n",self.decrypt_data(quadratic_form_encrypted))
                print("should be : ",np.multiply(self.decrypt_data(diff_encrypted),self.decrypt_data(covariance_inv_encrypted)),np.multiply((x-mean),covariance_inv))

                print(5*"**********"+"\n")
                print(5*"-------------------")
                print("diff_encrypted\n", self.decrypt_data(diff_encrypted))
                print("quadric form:\n", self.decrypt_data(quadratic_form_encrypted))
                # quadratic_form_encrypted =  diff_encrypted.dot(covariance_inv_encrypted)
                quadratic_form_encrypted =self.evaluate.multiply(quadratic_form_encrypted,diff_encrypted)
                print("result:\n" + 5 * "**********"+"\n", self.decrypt_data(quadratic_form_encrypted))
                print("should be : ",np.multiply(self.decrypt_data(quadratic_form_encrypted),self.decrypt_data(diff_encrypted)))
                print(5*"**********"+"\n")

                quadratic_form = self.decrypt_data(quadratic_form_encrypted)

                quadratic_form=np.array(quadratic_form[:len(responsibilities)])
                print(quadratic_form,responsibilities,covariance_det)

                log_probabilities[j, i] = np.log(responsibilities) - 0.5 * np.log(covariance_det) - 0.5 * quadratic_form
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
