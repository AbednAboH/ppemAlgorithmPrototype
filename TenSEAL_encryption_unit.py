from settings import *
import tenseal as ts
from numpy import power
class encryption:
    def __init__(self):
        self.context=ts.context(
            scheme=scheme,
            poly_modulus_degree=poly_modulus_degree,
            plain_modulus=plain_modulus,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            encryption_type=encryption_type, )
        # scale of ciphertext to use might not work on BFV model , surly works on CKKS
        self.context.global_scale = 2 ** 40
        # dot product key needed for the dot operation
        self.context.generate_galois_keys()
    """Encrypt the data using BFV encryption or CKKS"""
    def BFV_encrypt(self, data):
        data1 = power(data, 2) * 40
        tensorData = ts.bfv_tensor(self.context, data1)
        tensorData.square()
        return tensorData
    def CKKS_encrypt(self,data):
        data1 = power(data, 2) * 40
        tensorData = ts.bfv_tensor(self.context, data1)
        tensorData.square()

    def decrypt(self, encrypted_data):
        """Decrypt the data using BFV decryption"""
        return encrypted_data.decrypt().tolist()
