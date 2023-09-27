# Privacy Preserving Expectation maximization (PPEM) On Gaussian Mixture models
This is an implementation of our proposal of a new protocol that guarantees privacy to a certain degree
Supervised by Prof.Adi Akvia
Proposed by Abed Elrahman abo hussien and Michael Rodel

Some background on the project can be found ![here](Background.md).

Our proposal can be found ![here](our_approach.md).
# Custom usage of our implementation:

   Federated EM :
        To use This algorithm :
            1.import Server from CodeBase.Federated_EM_Server
            2.set the parameters:
                Parameters:
                    n :number of parameters.
                    max_iter :number of iterations
                    number_of_clustures :number of clusters to be associated with the data points
                    input :input of the PPEM algorithm when we use server client model
                    inputDimentions:the dimensions of the input array
                    epsilonExceleration:a boolean that enables/disables convergence with epsilons aid
                    eps:the value of epsilon that helps with the convergence criteria
                    plottingTools:if True the algorithm plots a gif of the EM process of the algorithm
                    clients:The number of clients expected /created
            3. run Server.Solve()
            4. the output should be 'pi, means, covariances, log_likelihoods, n_input, ticks, time_line'
   Privacy Preserving EM:
        To use This algorithm :
             1.import PPserver from CodeBase.PrivacyPreserving_EM_Server
                    2.set the parameters:
                        Parameters:
                            n :number of parameters.
                            max_iter :number of iterations
                            number_of_clustures :number of clusters to be associated with the data points
                            input :input of the PPEM algorithm when we use server client model
                            inputDimentions:the dimensions of the input array
                            epsilonExceleration:a boolean that enables/disables convergence with epsilons aid
                            eps:the value of epsilon that helps with the convergence criteria
                            plottingTools:if True the algorithm plots a gif of the EM process of the algorithm
                            clients:The number of clients expected /created
                    3. run Server.Solve()
                    4. the output should be 'pi, means, covariances, log_likelihoods, n_input, ticks, time_line'
   For more parameter tuning inspect the code.

# Tests and how to navigate them

  1.We performed tests on each algorithm with different sets of randomized data .
    each test is represented as 'algorithm_n{number of points}_k{clusters}_c{number of clients}'
    where algorithm is replaced by the algorithm PPEM or FEM.
    n- is the number of data points in the data set
    k- is the number of clusters (Gaussian Distributions)
    c- is the number of clients
* all tests were done on 2-D data , although the code is not bounded by it , changing the parameter 'inputDimentions' would give more dimentionality to the data,yet we reccomend disabling the plotting tools option.

  2.The tests were done on both algorithms:
       1. each individual drawing of the tests is in the 'Results/algorithm/algorithm_ni_kj_cm' folder with an excel that contains all relevant information of that test
       2. the Log likelihood test that compares both algorithms is in the 'Results/LikleyhoodComparisions' folder
       3. the parkinsons tests each are in both 'Results/algorithm/parkinsons' folder
       4. the excel that compares all data together is in the 'Results' folder named 'comparition data'
  
  3.The results include both images of the final clusters and CSV files that log The full process:
       a. images of the clusters are found in the 'Results\algorithm' file where algorithm is either PPEM or FEM
       b. CSV files that log each algorithms parameters are found in 'Results\algorithm\Charts' where algorithm is either PPEM or FEM
       c. images of the Loglilkelihood comparisions is found in 'Results\LikleyhoodComparisions' 
       d. a CSV file that contains the summary of all files found in subsection 'b' in 'Results\comparison_data'


    To reproduce our findings please run the TestsScript.py where all the folders would be created from scrach

    To reproduce Parkinsons or create a randomized data test you can use the main.py and then chose the appropriate options :

       Select the function you would like to do :
        1.Run on both algorithms
        2.PPEM
        3.Federated EM

       Then:
        1.use UCL's Parkinson DataBase that consists of 31 participants (23 infected) with 23 features
        2.custom parameters with randomized data
        3.enter a dataset with '.data'/'.csv'
       
       Then: 
	 follow the instructions given in the console
# Index
    To navigate through all the data ,test results and benchmarks check the 'Index.md' file


# Requirements
   found in the requirements.txt
    imageio==2.15.0
    matplotlib==3.3.4
    numpy==1.19.5
    pandas==2.1.1
    scikit_learn==1.3.1
    scipy==1.5.4
    seaborn==0.12.2
    self==2020.12.3
    tenseal==0.3.6

#

