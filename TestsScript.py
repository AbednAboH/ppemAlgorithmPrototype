from HelpingFunctions import compare_both_algorithms, compare_csv_files_by_nk
from PrivacyPreserving_EM_Server import PP_EM_Tests
from Federated_EM_Server import testsFEM


# test Federated EM on multiple variables
from Settings_Parameters import accepted_values

# print("-"*10+"\n Initiating Federated EM Tests\n"+"-"*10)
#
# # testsFEM()
#
# print("-"*10+"\nFederated EM Tests Ended\n"+"-"*10)
#
# # test PPEM on multiple variables
# print("-"*10+"\n Initiating Privacy Preserving EM Tests\n"+"-"*10)
#
# # PP_EM_Tests()
#
# print("-"*10+"\nPrivacy Preserving EM Tests Ended\n"+"-"*10)

# create a chart that compares both algorithms divergence time and number of iterations
# print(" Creating Results/comparison_data.csv file between both algorithms ")
# compare_both_algorithms('Results/FEM/Charts', 'Results/PPEM/Charts',
#                         'Results/comparison_data.csv', accepted_values)
# print("File created ,in Results/comparison_data.csv")
#
# # create log likelyhood plot for all input in those files
# print("plotting loge likleyhood for all results")
compare_csv_files_by_nk('Results/FEM/Charts', 'Results/PPEM/Charts',
                        'Results/LikleyhoodComparisions')

# compare both algorithms on the parkingsons data

# compare_csv_files_by_nk('Results/FEM/Parkingsons', 'Results/PPEM/Parkingsons',
#                         'Results/ParkingsonsLogLikelyhood')
