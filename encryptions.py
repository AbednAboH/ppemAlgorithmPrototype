import numpy as np
import seal

# Create an example 2D matrix
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Set up SEAL parameters
params = seal.EncryptionParameters(seal.scheme_type.bfv)
params.set_poly_modulus_degree(4096)
params.set_coeff_modulus(seal.CoeffModulus.BFVDefault(4096))
params.set_plain_modulus(seal.PlainModulus.Batching(4096, 20))

# Create SEAL context and key generator
context = seal.SEALContext(params)
keygen = seal.KeyGenerator(context)
public_key = keygen.create_public_key()
secret_key = keygen.secret_key()

# Create an encryptor and evaluator
encryptor = seal.Encryptor(context, public_key)
evaluator = seal.Evaluator(context)

# Create a BatchEncoder to encode and decode the 2D matrix
batch_encoder = seal.BatchEncoder(context)

# Get the slot count from the BatchEncoder
slot_count = batch_encoder.slot_count()

# Encode the matrix row by row
encoded_matrix = np.zeros((matrix.shape[0], slot_count), dtype=np.int64)
for i in range(matrix.shape[0]):
    row = matrix[i]
    encoded_row = batch_encoder.encode(row)
    encoded_matrix[i] = encoded_row

# Create a ciphertext object to store the encrypted result
encrypted_matrix = seal.Ciphertext()

# Encrypt the encoded matrix row by row
for i in range(encoded_matrix.shape[0]):
    plain_row = seal.Plaintext()
    plain_row.set_poly_modulus_degree(params.poly_modulus_degree())
    plain_row.set_coeff_modulus(params.coeff_modulus())
    plain_row.set_plain_modulus(params.plain_modulus())
    plain_row.set_value(encoded_matrix[i])
    encryptor.encrypt(plain_row, encrypted_matrix)

# Perform computations on the encrypted matrix using the evaluator
# ...

# Decrypt the encrypted matrix
decryptor = seal.Decryptor(context, secret_key)
plain_result = seal.Plaintext()
decryptor.decrypt(encrypted_matrix, plain_result)

# Decode the decrypted result row by row
decoded_matrix = np.zeros_like(encoded_matrix, dtype=np.int64)
for i in range(encoded_matrix.shape[0]):
    plain_row = seal.Plaintext()
    plain_row.set_poly_modulus_degree(params.poly_modulus_degree())
    plain_row.set_coeff_modulus(params.coeff_modulus())
    plain_row.set_plain_modulus(params.plain_modulus())
    plain_row.set_value(decoded_matrix[i])
    batch_encoder.decode(plain_row, decoded_matrix[i])

print("Original Matrix:")
print(matrix)
print("Decrypted Result:")
print(decoded_matrix)
