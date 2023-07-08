GA_MAXITER = 100
MAXSIGMA = 10  # todo figure actual starting value
MAXMU = 5  # todo figure actual starting value
import tenseal as ts

# encryption_parameters
scheme=ts.SCHEME_TYPE.CKKS
poly_modulus_degree=8192
coeff_mod_bit_sizes=[40, 21, 21, 21, 21, 21, 21, 40]
encryption_type=ts.ENCRYPTION_TYPE.SYMMETRIC

