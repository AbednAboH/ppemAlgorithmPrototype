GA_MAXITER = 100
MAXSIGMA = 10  # todo figure actual starting value
MAXMU = 5  # todo figure actual starting value
import tenseal as ts
import pandas as pd
# encryption_parameters
scheme=ts.SCHEME_TYPE.CKKS
poly_modulus_degree=4096
coeff_mod_bit_sizes=[40, 20, 40]
SCALE=2**40
encryption_type=ts.ENCRYPTION_TYPE.SYMMETRIC

# directories
PPEM_directory="Results/PPEM/"
PPEM_Preproccession_directory="Results/PPEM_Preprocessing/"
FEM_directory="Results/FEM/"
FEM_Preproccession_directory="Results/FEM_Preprocessing/"

# dataset properties
column_names = [
    "subject#",
    "age",
    "sex",
    "test_time",
    "motor_UPDRS",
    "total_UPDRS",
    "Jitter(%)",
    "Jitter(Abs)",
    "Jitter:RAP",
    "Jitter:PPQ5",
    "Jitter:DDP",
    "Shimmer",
    "Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "Shimmer:APQ11",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "PPE"
]
parkingson_data = pd.read_csv('DataSets/parkinsons.data',names=column_names)
