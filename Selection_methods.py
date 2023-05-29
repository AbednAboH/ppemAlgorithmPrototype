import tenseal as ts
import numpy as np

class PPEM:
    def __init__(self, poly_modulus_degree, plain_modulus):
        self.poly_modulus_degree = poly_modulus_degree
        self.plain_modulus = plain_modulus
        self.context = ts.context(
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
            plain_modulus=plain_modulus,
        )
        self.encoder = ts.encoder.Encoder(self.context)
        self.encryptor = ts.encryptor.Encryptor(self.context)
        self.decryptor = ts.decryptor.Decryptor(self.context)
        self.evaluator = ts.evaluator.Evaluator(self.context)

    def encrypt_data(self, data):
        plaintext = ts.plain_tensor(data)
        encrypted_data = self.encryptor.encrypt(plaintext)
        return encrypted_data

    def decrypt_data(self, encrypted_data):
        decrypted_data = self.decryptor.decrypt(encrypted_data)
        return decrypted_data

    def em_algorithm(self, data, n_iter):
        # Initialize means and covariances
        n_samples, n_features = data.shape
        means = np.random.randn(n_features)
        covariances = np.eye(n_features)
        responsibilities = np.zeros((n_samples, n_features))

        for _ in range(n_iter):
            # E-step: Calculate responsibilities
            for i in range(n_samples):
                x = data[i]
                x_encrypted = self.encrypt_data(x)
                means_encrypted = self.encrypt_data(means)
                covariances_encrypted = self.encrypt_data(covariances)

                diff_encrypted = self.evaluator.sub(x_encrypted, means_encrypted)
                quadratic_form_encrypted = self.evaluator.multiply(diff_encrypted, covariances_encrypted)
                quadratic_form_encrypted = self.evaluator.multiply(quadratic_form_encrypted, diff_encrypted)

                quadratic_form = self.decrypt_data(quadratic_form_encrypted)
                responsibilities[i] = np.exp(-0.5 * quadratic_form)

            # M-step: Update means and covariances
            total_responsibilities = np.sum(responsibilities, axis=0)
            means = np.sum(data * responsibilities, axis=0) / total_responsibilities
            covariances = np.diag(np.sum((data - means) ** 2 * responsibilities, axis=0) / total_responsibilities)

        return means, covariances

# Example usage
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
ppem = PPEM(poly_modulus_degree=4096, plain_modulus=40961)
means, covariances = ppem.em_algorithm(data, n_iter=10)
print("Means:", means)
print("Covariances:", covariances)
