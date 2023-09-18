GA_MAXITER = 100
MAXSIGMA = 10  # todo figure actual starting value
MAXMU = 5  # todo figure actual starting value
import tenseal as ts
import pandas as pd

# encryption_parameters
scheme = ts.SCHEME_TYPE.CKKS
poly_modulus_degree = 4096
coeff_mod_bit_sizes = [40, 20, 40]
SCALE = 2 ** 40
encryption_type = ts.ENCRYPTION_TYPE.SYMMETRIC

# directories
PPEM_directory = "Results/PPEM/"
PPEM_Preproccession_directory = "Results/PPEM_Preprocessing/"
FEM_directory = "Results/FEM/"
FEM_Preproccession_directory = "Results/FEM_Preprocessing/"

# Scrypt lines
CHOSEN_ALGO="1.Compare both algorithms\n2.PPEM\n3.Federated EM\n"
DATASET_SELECTION = "1.use UCL's Parkinson DataBase that consists of 31 participants (23 infected) with 23 features\n2.custom parameters with randomized data\n3.enter a dataset with '.data'/'.csv'\n"
CLUSTERS = "Enter the number of clusters you would like the data to be split to (2,100)\n"
ITERATIONS = "Enter the maximum iterations that you want for the algorithm (100,10000):\n"
NUM_CLIENTS="Enter the number of clients:\n"
PLOTTING = "press 1 to save the resulting plot 0 otherwise\n"
DIR = "would you like to save to a specific directory ? press 1/0 for yes or no\n"
DIRECTORY = "please enter the required directory: with '/' to separate between folders and with the name of the image\n"
NOTES_ON_DATA="this option is reserved for ucl data format and will not work otherwise \nthe format of the data should be as follows:\n first row would be the features and the rest would be the data itself\n"
SELECT_UCL_DATA="if the data is not as we have said in the above section please reformat it such that it follows the same protocol\n press 1 to continue,0 to the main menu\n"


#algorithm strings:
FEM="Federated Expectation Maximization"

# for CSV file filtering
accepted_values = ["ticks", "log_likelihoods", "Iterations"]

# UCL data format
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
parkingson_data = pd.read_csv('DataSets/parkinsons.data', names=column_names)

PERSESSION=0.00001

MAXITER=500